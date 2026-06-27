import pytorch_lightning as pl
import torch
from typing import Optional, Union, List, Dict, Any 
# from models.stable_audio.stable_audio_transformer import StableAudioDiTModel
# from models.stable_audio.stable_audio_modeling import StableAudioProjectionModel
# from models.wan_video.wan_video_transformer import WanTransformer3DModel
from models.sana_video.transformer_sana_video import SanaVideoTransformer3DModel
from models.audioldm.unet_2d_condition import UNet2DConditionModel
from models.dit.dit_dual_stream_clean import DualDiTBlockAdaLN, CrossDiTBlockAdaLN, AddFusionDiTBlockAdaLN
from diffusers.models import AutoencoderDC, AutoencoderOobleck, AutoencoderKLWan, AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler, KarrasDiffusionSchedulers, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, Gemma2Model, GemmaTokenizer, GemmaTokenizerFast, ClapTextModelWithProjection, RobertaTokenizer, RobertaTokenizerFast, SpeechT5HifiGan
from diffusers.models.embeddings import get_1d_rotary_pos_embed
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm
from diffusers.utils import export_to_video
import torchvision.io as tvio
import torchvision.transforms as transforms
import torchaudio
import torchvision
import soundfile as sf
import regex as re
import ftfy
import html
import inspect
import yaml
from moviepy import VideoClip, AudioArrayClip
import numpy as np
import time
import torch.nn as nn
from functools import partial
from datetime import datetime
import os
import torch.nn.functional as F

class BasePipeline(pl.LightningModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.load_path_config = self.config['load_path_config']
        self.train_config = self.config['train_config']
        self.model_archi_config = self.config['model_archi_config']
        self.sample_config = self.config['sample_config']
        self.weight_dtype = torch.bfloat16
        get_huggingface_model_from_local_dir = lambda ckpt_dir, model: \
            model.from_pretrained(
                ckpt_dir, local_files_only=True,    # From pretrained
                ignore_mismatched_sizes=False,      # Strict for pretrained model structure
                # use_safetensors=True,               # Safe tensor
                torch_dtype=self.weight_dtype,      # bfloat16
            )
        self.model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ## Load Model Part
        ckpt_dir_audio_diffusion = self.load_path_config['audio_model_path']['ckpt_dir_audio_diffusion']
        ckpt_dir_audio_vae = self.load_path_config['audio_model_path']['ckpt_dir_audio_vae']
        audio_pipeline_scheduler_config = self.load_path_config['audio_model_path']['audio_pipeline_scheduler_config']
        audio_text_encoder = self.load_path_config['audio_model_path']['audio_text_encoder']
        audio_tokenizer = self.load_path_config['audio_model_path']['audio_tokenizer']
        audio_vocoder_path = self.load_path_config['audio_model_path']['audio_vocoder_path']
        ckpt_dir_video_vae = self.load_path_config['video_model_path']['ckpt_dir_video_vae']
        ckpt_dir_video_diffusion = self.load_path_config['video_model_path']['ckpt_dir_video_diffusion']
        video_pipeline_scheduler_config = self.load_path_config['video_model_path']['video_pipeline_scheduler_config']
        video_text_encoder = self.load_path_config['video_model_path']['video_text_encoder']
        video_tokenizer = self.load_path_config['video_model_path']['video_tokenizer']
        self.audio_diffusion = get_huggingface_model_from_local_dir(ckpt_dir_audio_diffusion, UNet2DConditionModel)
        self.audio_vae = get_huggingface_model_from_local_dir(ckpt_dir_audio_vae, AutoencoderKL)
        # self.audio_text_encoder = get_huggingface_model_from_local_dir(audio_text_encoder, T5EncoderModel)
        self.audio_tokenizer = AutoTokenizer.from_pretrained(audio_tokenizer)
        self.audio_pipeline_scheduler = get_huggingface_model_from_local_dir(audio_pipeline_scheduler_config, DDIMScheduler)
        self.audio_vocoder = SpeechT5HifiGan.from_pretrained(audio_vocoder_path)
        self.sample_audio_length_s = self.config.get('sample_audio_length_s', 5.40)
        self.sample_audio_sampling_rate = self.config.get('sample_audio_sampling_rate', 44100)
        self.video_diffusion = get_huggingface_model_from_local_dir(ckpt_dir_video_diffusion, SanaVideoTransformer3DModel)
        self.video_vae = get_huggingface_model_from_local_dir(ckpt_dir_video_vae, AutoencoderKLWan)
        self.video_tokenizer = AutoTokenizer.from_pretrained(video_tokenizer)
        self.video_pipeline_scheduler = get_huggingface_model_from_local_dir(video_pipeline_scheduler_config, DPMSolverMultistepScheduler)
        ## Sample Config Part
        self.sample_video_fps = self.sample_config.get('sample_video_fps', 15)
        self.sample_video_num_frames = self.sample_config.get('sample_video_num_frames', 81)
        self.sample_video_height = self.sample_config.get('sample_video_height', 480)
        self.sample_video_width = self.sample_config.get('sample_video_width', 832)
        self.num_inference_steps = self.sample_config.get('num_inference_steps', 50)
        self.sample_video_guidance_scale = self.sample_config.get('sample_video_guidance_scale', 6)
        self.num_samples_per_prompt = self.sample_config.get('num_samples_per_prompt', 1)
        self.inference_seed = self.sample_config.get('inference_seed', 0)
        self.audio_guidance_scale = self.sample_config.get('audio_guidance_scale', 7)
        self.video_save_path = self.sample_config.get('video_save_path', "save_videos")
        self.video_vae_scale_factor_temporal = 2 ** sum(self.video_vae.temperal_downsample) if getattr(self, "video_vae", None) else 4
        self.audio_vae_scale_factor = 2 ** (len(self.audio_vae.config.block_out_channels) - 1) if getattr(self, "audio_vae", None) else 8
        self.video_vae_scale_factor_spatial = 2 ** len(self.video_vae.temperal_downsample) if getattr(self, "video_vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.video_vae_scale_factor_spatial)
        self.negative_prompt = self.sample_config.get('negative_prompt', "Low quality")
        self.use_condition_embedding_cache = self.train_config.get('cache_config', {}).get('use_condition_embedding_cache', True)
        if not self.use_condition_embedding_cache:
            print(audio_text_encoder, video_text_encoder)
            self.audio_text_encoder = get_huggingface_model_from_local_dir(audio_text_encoder, ClapTextModelWithProjection)
            self.video_text_encoder = get_huggingface_model_from_local_dir(video_text_encoder, Gemma2Model)
        print("Successfully loaded model")
        
    def forward(self):
        pass

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        mel_spectrogram = mel_spectrogram.to(self.audio_vocoder.dtype)
        waveform = self.audio_vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()  
        return waveform

    def prepare_audio_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.audio_vae_scale_factor,
            int(self.audio_vocoder.config.model_in_dim) // self.audio_vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.audio_pipeline_scheduler.init_noise_sigma
        return latents

    def prepare_video_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.video_vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.video_vae_scale_factor_spatial,
            int(width) // self.video_vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents
    
    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # prepare complex human instruction
        if not complex_human_instruction:
            max_length_all = max_sequence_length
        else:
            chi_prompt = "\n".join(complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_prompt_tokens = len(self.video_tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + max_sequence_length - 2

        text_inputs = self.video_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = self.video_text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask

    def encode_video_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
    ):
        if device is None:
            device = self._execution_device

        if self.video_text_encoder is not None:
            dtype = self.video_text_encoder.dtype
        else:
            dtype = None
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "right"

        # See Section 3.1. of the paper.
        max_length = max_sequence_length
        select_index = [0] + list(range(-max_length + 1, 0))

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
            )

            prompt_embeds = prompt_embeds[:, select_index]
            prompt_attention_mask = prompt_attention_mask[:, select_index]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=False,
            )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None
        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
             
    def encode_audio_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.audio_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.audio_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = self.audio_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            #     text_input_ids, untruncated_ids
            # ):
            #     removed_text = self.tokenizer.batch_decode(
            #         untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            #     )
            #     logger.warning(
            #         "The following part of your input was truncated because CLAP can only handle sequences up to"
            #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #     )

            prompt_embeds = self.audio_text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            prompt_embeds = F.normalize(prompt_embeds, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.audio_text_encoder.dtype, device=device)

        (
            bs_embed,
            seq_len,
        ) = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.audio_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.audio_text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds.text_embeds
            # additional L_2 normalization over each hidden-state
            negative_prompt_embeds = F.normalize(negative_prompt_embeds, dim=-1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.audio_text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_waveforms_per_prompt, seq_len)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    def denoising(self):
        pass

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.video_pipeline_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.video_pipeline_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        accepts_eta = "eta" in set(inspect.signature(self.audio_pipeline_scheduler.step).parameters.keys())
        extra_step_kwargs_a = {}
        if accepts_eta:
            extra_step_kwargs_a["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.audio_pipeline_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs_a["generator"] = generator

        return extra_step_kwargs, extra_step_kwargs_a   

class Seperate_infer(BasePipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        # Freeze.
        self.video_frame_transform = transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x/255 * 2 - 1),
            torchvision.transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            torchvision.transforms.Resize((480, 832), antialias=True),  
        ])
        self.duration = 5.40
        self.audio_process_config = {
            "sampling_rate": 44100,
            "min_waveform_length": 100,
        }
        self.video_process_config = {
            "target_frame_nums": 81,
            "target_fps": 15,
            "vae_scale_factor": 4,
            "video_h": 480,
            "video_w": 832,
        }
        for _module in [self.video_vae, self.video_diffusion, self.audio_vae, self.audio_diffusion]:
            _module.requires_grad_(False)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.requires_grad_(False)
            self.audio_text_encoder.requires_grad_(False)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.unconditional_prob = self.train_config.get('unconditional_prob', 0.1)
        assert 0.0 <= self.unconditional_prob <= 1.0, "Unconditional_prob should be in [0.0, 1.0]"
        
        self.video_diffusion.to(self.weight_dtype)
        self.audio_diffusion.to(self.weight_dtype)
        self.video_vae.to(self.weight_dtype)
        self.audio_vae.to(self.weight_dtype)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.to(self.weight_dtype)
            self.audio_text_encoder.to(self.weight_dtype)
        self.pre_process = self.train_config['cache_config'].get('pre_process', False)
        weight_dtype_str = self.train_config.get('weight_dtype', 'torch.bfloat16')
  
        
    def predict_step(self, video_prompt, audio_prompt, video_negative_prompt, audio_negative_prompt, save_name='test'):
        batch_size = 1
        if self.inference_seed is not None:
            rand_seed = self.inference_seed
        else:
            rand_seed = torch.randint(0, 100000, (1,),).item()
        generator = torch.Generator(device=self.model_device).manual_seed(rand_seed)
        if video_negative_prompt is None:
            video_negative_prompt = ["Low quality."]
        if audio_negative_prompt is None:
            audio_negative_prompt = ["Low quality."]
        self.video_text_encoder.to(self.model_device)
        self.audio_text_encoder.to(self.model_device)
        self.video_diffusion.to(self.model_device)
        self.audio_diffusion.to(self.model_device)
        self.video_vae.to(self.model_device)
        self.audio_vae.to(self.model_device)
        self.audio_vocoder.to(self.model_device)
        video_tensor, audio_tensor = self.denoising(
            video_prompt_text=video_prompt,
            audio_prompt_text=audio_prompt,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
            video_height=self.sample_video_height,
            video_width=self.sample_video_width,
            video_num_frames=self.sample_video_num_frames,
            generator=generator,
        )
        # video_tensor shape (batch_size, frame_nums, height, width, channel)
        # Log video
        for sample_idx in range(self.num_samples_per_prompt):
            cur_video = video_tensor[sample_idx]
            cur_video_path = os.path.join(self.video_save_path, "{}.mp4".format(save_name))
            video_np = cur_video  # e.g. shape (T, H, W, C)
            video_np = (video_np * 255).astype(np.uint8)  # 转成uint8，范围0-255，moviepy需要
            fps = self.sample_video_fps
            duration = video_np.shape[0] / fps
            def make_frame(t):
                frame_index = min(int(t * fps), video_np.shape[0] - 1)
                return video_np[frame_index]
            video_clip = VideoClip(make_frame, duration=duration)
            # 2. 处理音频数据：audio_tensor[0] 维度为 (channels, samples)
            audio_np = audio_tensor[sample_idx].T.float().cpu().numpy()  # 转成 numpy, shape (samples, channels)
            if audio_np.ndim == 1:
                # 从 (86400,) 变为 (86400, 2)
                audio_np = audio_np.reshape(-1, 1) 
                audio_np = np.repeat(audio_np, 2, axis=1)  # 复制一份到第二个通道 
            
            print(audio_np.shape)
            sampling_rate = 16000
            audio_clip = AudioArrayClip(audio_np, fps=sampling_rate)
            print("Audio Duration:", audio_clip.duration)
            video_clip = video_clip.with_audio(audio_clip)
            video_clip.write_videofile(cur_video_path, fps=fps, codec="libx264", audio_codec="aac")
        return video_tensor, audio_tensor

    @torch.no_grad()
    def denoising(
        self,
        video_prompt_text: Union[str, List[str]] = None,
        audio_prompt_text: Union[str, List[str]] = None,
        video_negative_prompt: Union[str, List[str]] = None,
        audio_negative_prompt: Union[str, List[str]] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        video_num_frames: int = 81,
        video_latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        num_samples_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_text_max_sequence_length: int = 1024,
        video_guidance_scale: float = 6.0,
        audio_guidance_scale: float = 7,
        audio_end_in_s: float = 5.40
    ):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        

        # 2. Default call parameters
        video_do_classifier_free_guidance = video_guidance_scale > 1.0
        audio_do_classifier_free_guidance = audio_guidance_scale > 1.0
        if video_prompt_text is not None and isinstance(video_prompt_text, str):
            batch_size = 1
        elif video_prompt_text is not None and isinstance(video_prompt_text, list):
            batch_size = len(video_prompt_text)
        vocoder_upsample_factor = np.prod(self.audio_vocoder.config.upsample_rates) / self.audio_vocoder.config.sampling_rate
        # waveform_length = int(self.audio_diffusion.config.sample_size)
        audio_length_in_s = audio_end_in_s

        audio_height = int(audio_length_in_s / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.audio_vocoder.config.sampling_rate)
        # 3. Encode input prompt
        (
            video_prompt_embeds,
            video_prompt_attention_mask,
            video_negative_prompt_embeds,
            video_negative_prompt_attention_mask,
        ) = self.encode_video_prompt(
            video_prompt_text,
            video_do_classifier_free_guidance,
            negative_prompt=video_negative_prompt,
            device=device
        )
         # Optional: Concatenate the prompt embeds for classifier free guidance
        if video_do_classifier_free_guidance:
            video_prompt_embeds = torch.cat([video_negative_prompt_embeds, video_prompt_embeds], dim=0)
            video_prompt_attention_mask = torch.cat([video_negative_prompt_attention_mask, video_prompt_attention_mask], dim=0) 
        audio_prompt_embeds = self.encode_audio_prompt(
            prompt=audio_prompt_text,
            device=device,
            do_classifier_free_guidance=audio_do_classifier_free_guidance,
            negative_prompt=audio_negative_prompt
        )
        # 4. Prepare timesteps
        self.video_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        video_timesteps = self.video_pipeline_scheduler.timesteps
        self.audio_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        audio_timesteps = self.audio_pipeline_scheduler.timesteps
        # 5. Prepare latents
        audio_latents = self.prepare_audio_latents(
            batch_size * num_samples_per_prompt, 
            self.audio_diffusion.config.in_channels,
            audio_height,
            audio_prompt_embeds.dtype, 
            device, 
            generator,           
            None,
        )    
        
        video_latents = self.prepare_video_latents(
            batch_size,
            self.video_diffusion.config.in_channels,
            video_height,
            video_width,
            video_num_frames,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        # 6. Prepare extra step kwargs
        extra_step_kwargs_v, extra_step_kwargs_a = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Denoising loop
        old_pred_original_sample = None
        # Audio Latent [1, 8 , 67, 8]
        # Video Latent [1, 16, 21, 60 , 104]
        for i, vt in tqdm(enumerate(video_timesteps), total=len(video_timesteps)):
            # 8.1 Prepare audio latents for the step
            at = audio_timesteps[i]
            audio_latents_model_input = torch.cat([audio_latents] * 2) if audio_do_classifier_free_guidance else audio_latents
            audio_latents_model_input = self.audio_pipeline_scheduler.scale_model_input(audio_latents_model_input, at)
            # 8.2 Prepare video latents for the step
            video_latent_model_input = torch.cat([video_latents] * 2) if video_do_classifier_free_guidance else video_latents
            # 8.2 Prepare input kwargs for video and audio diffusion models
            video_timestep = vt.expand(video_latent_model_input.shape[0])
            video_diff_input_kwargs = { 
                "hidden_states": video_latent_model_input.to(self.video_diffusion.dtype),
                "encoder_hidden_states": video_prompt_embeds.to(self.video_diffusion.dtype),
                "encoder_attention_mask": video_prompt_attention_mask.to(self.video_diffusion.dtype),
                "timestep": video_timestep,
                "return_dict": False,
            }

            audio_diff_input_kwargs = {
                "sample": audio_latents_model_input,
                "timestep": at.unsqueeze(0),
                "encoder_hidden_states": None,
                "class_labels": audio_prompt_embeds.to(self.audio_diffusion.dtype),
            }
            video_noise_pred = self.video_diffusion(**video_diff_input_kwargs)[0].float()
            audio_noise_pred = self.audio_diffusion(**audio_diff_input_kwargs).sample
            # perform video guidance
            if video_do_classifier_free_guidance:
                video_noise_pred_uncond, video_noise_pred_text = video_noise_pred.chunk(2)
                video_noise_pred = video_noise_pred_uncond + video_guidance_scale * (video_noise_pred_text - video_noise_pred_uncond)       
        
            # perform audio guidance
            if audio_do_classifier_free_guidance:
                audio_noise_pred_uncond, audio_noise_pred_text = audio_noise_pred.chunk(2)
                audio_noise_pred = audio_noise_pred_uncond + audio_guidance_scale * (audio_noise_pred_text - audio_noise_pred_uncond)
            # Shape of Video Noist Pred: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Noist Pred: (batch_size, 64, 104)
            # Audio Latent [1, 8 , 67, 8]
            # Video Latent [1, 16, 21, 60 , 104]
            # compute the previous noisy sample x_t -> x_t-1

            audio_latents = self.audio_pipeline_scheduler.step(audio_noise_pred, at, audio_latents, **extra_step_kwargs_a).prev_sample
            video_latents = self.video_pipeline_scheduler.step(video_noise_pred, vt, video_latents, return_dict=False)[0]
            # Shape of Video Latents: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Latents: (batch_size, 64, 104)
        # 9. Post-processing
        ## Process video
        video_latents = video_latents.to(self.video_vae.dtype)
        video_latents_mean = (
            torch.tensor(self.video_vae.config.latents_mean)
            .view(1, self.video_vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        video_latents_std = 1.0 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )
        video_latents = video_latents / video_latents_std + video_latents_mean
        video = self.video_vae.decode(video_latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type="np")
        ## Process audio
        audio_latents = audio_latents.to(self.audio_vae.dtype)
        audio_latents = 1 / self.audio_vae.config.scaling_factor * audio_latents
        mel_spectrogram = self.audio_vae.decode(audio_latents).sample
        # mel_spectrogram shape torch.Size([1, 1, 268, 32])
         # mel_spectrogram shape: (batch_size, 1, mel_bins, time_steps)
        mel_spectrogram = mel_spectrogram.to(self.audio_vocoder.dtype)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        # audio = audio[:, 0, waveform_start:waveform_end]
        return video, audio

class JointInfer(BasePipeline):
    def __init__(self, config_path):
        super().__init__(config_path)
        # Freeze.
        self.video_frame_transform = transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x/255 * 2 - 1),
            torchvision.transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            torchvision.transforms.Resize((480, 832), antialias=True),  
        ])
        self.duration = 5.40
        self.audio_process_config = {
            "sampling_rate": 44100,
            "min_waveform_length": 100,
        }
        self.video_process_config = {
            "target_frame_nums": 81,
            "target_fps": 15,
            "vae_scale_factor": 4,
            "video_h": 480,
            "video_w": 832,
        }
        for _module in [self.video_vae, self.video_diffusion, self.audio_vae, self.audio_diffusion]:
            _module.requires_grad_(False)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.requires_grad_(False)
            self.audio_text_encoder.requires_grad_(False)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)
        self.unconditional_prob = self.train_config.get('unconditional_prob', 0.1)
        assert 0.0 <= self.unconditional_prob <= 1.0, "Unconditional_prob should be in [0.0, 1.0]"
        
        self.video_diffusion.to(self.weight_dtype)
        self.audio_diffusion.to(self.weight_dtype)
        self.video_vae.to(self.weight_dtype)
        self.audio_vae.to(self.weight_dtype)
        if not self.use_condition_embedding_cache:
            self.video_text_encoder.to(self.weight_dtype)
            self.audio_text_encoder.to(self.weight_dtype)
        self.pre_process = self.train_config['cache_config'].get('pre_process', False)
  
        
    def predict_step(self, video_prompt, audio_prompt, video_negative_prompt, audio_negative_prompt, save_name='test'):
        batch_size = 1
        if self.inference_seed is not None:
            rand_seed = self.inference_seed
        else:
            rand_seed = torch.randint(0, 100000, (1,),).item()
        generator = torch.Generator(device=self.model_device).manual_seed(rand_seed)
        if video_negative_prompt is None:
            video_negative_prompt = ["Low quality."]
        if audio_negative_prompt is None:
            audio_negative_prompt = ["Low quality."]
        self.video_text_encoder.to(self.model_device)
        self.audio_text_encoder.to(self.model_device)
        self.video_diffusion.to(self.model_device)
        self.audio_diffusion.to(self.model_device)
        self.video_vae.to(self.model_device)
        self.audio_vae.to(self.model_device)
        self.audio_vocoder.to(self.model_device)
        video_tensor, audio_tensor = self.denoising(
            video_prompt_text=video_prompt,
            audio_prompt_text=audio_prompt,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
            video_height=self.sample_video_height,
            video_width=self.sample_video_width,
            video_num_frames=self.sample_video_num_frames,
            generator=generator,
        )
        # video_tensor shape (batch_size, frame_nums, height, width, channel)
        # Log video
        for sample_idx in range(self.num_samples_per_prompt):
            cur_video = video_tensor[sample_idx]
            cur_video_path = os.path.join(self.video_save_path, "{}.mp4".format(save_name))
            video_np = cur_video  # e.g. shape (T, H, W, C)
            video_np = (video_np * 255).astype(np.uint8)  # 转成uint8，范围0-255，moviepy需要
            fps = self.sample_video_fps
            duration = video_np.shape[0] / fps
            def make_frame(t):
                frame_index = min(int(t * fps), video_np.shape[0] - 1)
                return video_np[frame_index]
            video_clip = VideoClip(make_frame, duration=duration)
            # 2. 处理音频数据：audio_tensor[0] 维度为 (channels, samples)
            audio_np = audio_tensor[sample_idx].T.float().cpu().numpy()  # 转成 numpy, shape (samples, channels)
            if audio_np.ndim == 1:
                # 从 (86400,) 变为 (86400, 2)
                audio_np = audio_np.reshape(-1, 1) 
                audio_np = np.repeat(audio_np, 2, axis=1)  # 复制一份到第二个通道 
            
            print(audio_np.shape)
            sampling_rate = 16000
            audio_clip = AudioArrayClip(audio_np, fps=sampling_rate)
            print("Audio Duration:", audio_clip.duration)
            video_clip = video_clip.with_audio(audio_clip)
            video_clip.write_videofile(cur_video_path, fps=fps, codec="libx264", audio_codec="aac")
        return video_tensor, audio_tensor

    @torch.no_grad()
    def denoising(
        self,
        video_prompt_text: Union[str, List[str]] = None,
        audio_prompt_text: Union[str, List[str]] = None,
        video_negative_prompt: Union[str, List[str]] = None,
        audio_negative_prompt: Union[str, List[str]] = None,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        video_num_frames: int = 81,
        video_latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        num_samples_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_text_max_sequence_length: int = 1024,
        video_guidance_scale: float = 6.0,
        audio_guidance_scale: float = 7,
        audio_end_in_s: float = 5.40
    ):
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        

        # 2. Default call parameters
        video_do_classifier_free_guidance = video_guidance_scale > 1.0
        audio_do_classifier_free_guidance = audio_guidance_scale > 1.0
        if video_prompt_text is not None and isinstance(video_prompt_text, str):
            batch_size = 1
        elif video_prompt_text is not None and isinstance(video_prompt_text, list):
            batch_size = len(video_prompt_text)
        vocoder_upsample_factor = np.prod(self.audio_vocoder.config.upsample_rates) / self.audio_vocoder.config.sampling_rate
        # waveform_length = int(self.audio_diffusion.config.sample_size)
        audio_length_in_s = audio_end_in_s

        audio_height = int(audio_length_in_s / vocoder_upsample_factor)
        original_waveform_length = int(audio_length_in_s * self.audio_vocoder.config.sampling_rate)
        # 3. Encode input prompt
        (
            video_prompt_embeds,
            video_prompt_attention_mask,
            video_negative_prompt_embeds,
            video_negative_prompt_attention_mask,
        ) = self.encode_video_prompt(
            video_prompt_text,
            video_do_classifier_free_guidance,
            negative_prompt=video_negative_prompt,
            device=device
        )
         # Optional: Concatenate the prompt embeds for classifier free guidance
        if video_do_classifier_free_guidance:
            video_prompt_embeds = torch.cat([video_negative_prompt_embeds, video_prompt_embeds], dim=0)
            video_prompt_attention_mask = torch.cat([video_negative_prompt_attention_mask, video_prompt_attention_mask], dim=0) 
        audio_prompt_embeds = self.encode_audio_prompt(
            prompt=audio_prompt_text,
            device=device,
            do_classifier_free_guidance=audio_do_classifier_free_guidance,
            negative_prompt=audio_negative_prompt
        )
        # 4. Prepare timesteps
        self.video_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        video_timesteps = self.video_pipeline_scheduler.timesteps
        self.audio_pipeline_scheduler.set_timesteps(num_inference_steps, device=device)
        audio_timesteps = self.audio_pipeline_scheduler.timesteps
        # 5. Prepare latents
        audio_latents = self.prepare_audio_latents(
            batch_size * num_samples_per_prompt, 
            self.audio_diffusion.config.in_channels,
            audio_height,
            audio_prompt_embeds.dtype, 
            device, 
            generator,           
            None,
        )    
        
        video_latents = self.prepare_video_latents(
            batch_size,
            self.video_diffusion.config.in_channels,
            video_height,
            video_width,
            video_num_frames,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        # 6. Prepare extra step kwargs
        extra_step_kwargs_v, extra_step_kwargs_a = self.prepare_extra_step_kwargs(generator, eta)
        # 7. Denoising loop
        old_pred_original_sample = None
        # Audio Latent [1, 8 , 67, 8]
        # Video Latent [1, 16, 21, 60 , 104]
        for i, vt in tqdm(enumerate(video_timesteps), total=len(video_timesteps)):
            # 8.1 Prepare audio latents for the step
            at = audio_timesteps[i]
            audio_latents_model_input = torch.cat([audio_latents] * 2) if audio_do_classifier_free_guidance else audio_latents
            audio_latents_model_input = self.audio_pipeline_scheduler.scale_model_input(audio_latents_model_input, at)
            # 8.2 Prepare video latents for the step
            video_latent_model_input = torch.cat([video_latents] * 2) if video_do_classifier_free_guidance else video_latents
            # 8.2 Prepare input kwargs for video and audio diffusion models
            video_timestep = vt.expand(video_latent_model_input.shape[0])
            video_diff_input_kwargs = { 
                "hidden_states": video_latent_model_input.to(self.video_diffusion.dtype),
                "encoder_hidden_states": video_prompt_embeds.to(self.video_diffusion.dtype),
                "encoder_attention_mask": video_prompt_attention_mask.to(self.video_diffusion.dtype),
                "timestep": video_timestep,
                "return_dict": False,
            }

            audio_diff_input_kwargs = {
                "sample": audio_latents_model_input,
                "timestep": at.unsqueeze(0),
                "encoder_hidden_states": None,
                "class_labels": audio_prompt_embeds.to(self.audio_diffusion.dtype),
            }
            video_noise_pred = self.video_diffusion(**video_diff_input_kwargs)[0].float()
            audio_noise_pred = self.audio_diffusion(**audio_diff_input_kwargs).sample
            # perform video guidance
            if video_do_classifier_free_guidance:
                video_noise_pred_uncond, video_noise_pred_text = video_noise_pred.chunk(2)
                video_noise_pred = video_noise_pred_uncond + video_guidance_scale * (video_noise_pred_text - video_noise_pred_uncond)       
        
            # perform audio guidance
            if audio_do_classifier_free_guidance:
                audio_noise_pred_uncond, audio_noise_pred_text = audio_noise_pred.chunk(2)
                audio_noise_pred = audio_noise_pred_uncond + audio_guidance_scale * (audio_noise_pred_text - audio_noise_pred_uncond)
            # Shape of Video Noist Pred: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Noist Pred: (batch_size, 64, 104)
            # Audio Latent [1, 8 , 67, 8]
            # Video Latent [1, 16, 21, 60 , 104]
            # compute the previous noisy sample x_t -> x_t-1

            audio_latents = self.audio_pipeline_scheduler.step(audio_noise_pred, at, audio_latents, **extra_step_kwargs_a).prev_sample
            video_latents = self.video_pipeline_scheduler.step(video_noise_pred, vt, video_latents, return_dict=False)[0]
            # Shape of Video Latents: (batch_size, 16, 21 , 60, 104)
            # Shape of Audio Latents: (batch_size, 64, 104)
        # 9. Post-processing
        ## Process video
        video_latents = video_latents.to(self.video_vae.dtype)
        video_latents_mean = (
            torch.tensor(self.video_vae.config.latents_mean)
            .view(1, self.video_vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        video_latents_std = 1.0 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )
        video_latents = video_latents / video_latents_std + video_latents_mean
        video = self.video_vae.decode(video_latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type="np")
        ## Process audio
        audio_latents = audio_latents.to(self.audio_vae.dtype)
        audio_latents = 1 / self.audio_vae.config.scaling_factor * audio_latents
        mel_spectrogram = self.audio_vae.decode(audio_latents).sample
        # mel_spectrogram shape torch.Size([1, 1, 268, 32])
         # mel_spectrogram shape: (batch_size, 1, mel_bins, time_steps)
        mel_spectrogram = mel_spectrogram.to(self.audio_vocoder.dtype)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]
        # audio = audio[:, 0, waveform_start:waveform_end]
        return video, audio

    def process_video(self, video_paths):
        video_frames = []
        for video_path in video_paths:
            video_tuple = torchvision.io.read_video(video_path, pts_unit='sec')
            video_raw_frame_num = len(video_tuple[0])
            video_raw_fps = video_tuple[2]['video_fps']
            video_raw_duration = video_raw_frame_num / video_raw_fps
            video_tensor = video_tuple[0]
            video_tensor = self.video_frame_transform(video_tensor)         
            ''' Sample frames as the same as the audio part and Pad'''
            if video_raw_frame_num <= self.video_process_config["target_frame_nums"]:
                temp_video = video_tensor.repeat(10, 1, 1, 1)
                sample_videos = temp_video[:self.video_process_config["target_frame_nums"]]
            else:
                sample_videos = video_tensor[:self.video_process_config["target_frame_nums"]]
            video_frames.append(sample_videos.permute(1, 0, 2, 3))
        video_frames = torch.stack(video_frames, dim=0)
        return video_frames
    
    def process_audio(self, audio_paths):
        waveforms = []
        for audio_path in audio_paths:
            waveform, sr = torchaudio.load(audio_path)
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_process_config["sampling_rate"])
            waveform = waveform[0, ...].float()
            # Validate the waveform length
            waveform_length = waveform.shape[-1]
            ''' Random segment and pad '''
            target_length = int(self.audio_process_config["sampling_rate"] * self.duration)
            if waveform_length <= target_length:
                temp_wav = waveform.repeat(10)
                sample_waveforms = temp_wav[:target_length]
            else: 
                cur_waveform_1 = waveform[:target_length]
                sample_waveforms = cur_waveform_1
            waveforms.append(sample_waveforms)
        waveforms = torch.stack(waveforms, dim=0)
        return waveforms

    def forward(self):
        return super().forward()
    
    def retrieve_latents(
        self, encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
    ):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        else:
            raise AttributeError("Could not access latents of provided encoder_output")


    def train_step(self, batch):
        # input process
        video_path, video_prompt, audio_prompt = self.get_input(batch)
        video_frame = self.process_video(video_path)
        waveform = self.process_audio(video_path)
        batch_size = video_frame.shape[0]
        device = self.model_device

        self.audio_vae.to(device, self.weight_dtype)
        self.video_vae.to(device, self.weight_dtype)
        if self.inference_seed is not None:
            rand_seed = self.inference_seed
        else:
            rand_seed = torch.randint(0, 100000, (1,),).item()
        generator = torch.Generator(device=device).manual_seed(rand_seed)
        video_frame = video_frame.to(device, self.weight_dtype)
        with torch.no_grad():
            video_latents = self.retrieve_latents(self.video_vae.encode(video_frame), generator, 'sample').unbind(0)
        del video_frame
        video_latents_list = []
        for i in range(len(video_latents)):
            video_latents_list.append(video_latents[i].unsqueeze(0))
        video_latents = torch.cat(video_latents_list, dim=0)
        ## 2. Noramalize video latents
        video_latents_mean = (
            torch.tensor(self.video_vae.config.latents_mean)
            .view(1, self.video_vae.config.z_dim, 1, 1, 1)
            .to(video_latents.device, video_latents.dtype)
        )
        video_latents_std = 1.0 / torch.tensor(self.video_vae.config.latents_std).view(1, self.video_vae.config.z_dim, 1, 1, 1).to(
            video_latents.device, video_latents.dtype
        )
        video_latents = (video_latents - video_latents_mean) * video_latents_std
        video_latents = video_latents.detach()
        ## 3. Encoder Text prompt
        (
            video_prompt_embeds,
            video_prompt_attention_mask,
            video_negative_prompt_embeds,
            video_negative_prompt_attention_mask,
        ) = self.encode_video_prompt(
            video_prompt,
            True,
            negative_prompt=[self.negative_prompt]*batch_size,
            device=device
        )
        t = self.rng.draw(batch_size)[:, 0].to(self.model_device, self.weight_dtype)
        video_timesteps = t * (self.video_pipeline_scheduler.config.num_train_timesteps - 1)
        video_timesteps = video_timesteps.long()
        ## 5. Add noise to video latents
        video_noise = torch.randn_like(video_latents, dtype=self.weight_dtype)
        video_sigma = (video_timesteps / 1000).to(video_latents.dtype).view(-1, 1, 1, 1, 1)
        video_latents_noisy = (1 - video_sigma) * video_latents + video_noise * video_sigma
        '''For Audio Part'''

        ### 1. Convert wav to latent space
        audio_vae_weight_dtype = self.audio_vae.encoder.conv1.weight.dtype

        '''Forward and Loss'''
        video_input_kwargs = {
            "hidden_states": video_latents_noisy.to(self.video_diffusion.dtype),
            "encoder_hidden_states": video_prompt_embeds.to(self.video_diffusion.dtype),
            "encoder_attention_mask": video_prompt_attention_mask.to(self.video_diffusion.dtype),
            "timestep": video_timesteps,
            "return_dict": False,
        }

        audio_input_kwargs = {
            "sample": None,
            "timestep": None,
            "encoder_hidden_states": None,
            "class_labels": None,
        }

        video_pre, audio_pre = self.forward(
            video_input_kwargs, 
            audio_input_kwargs,
            self.rope_pos_embeds_1d_video,
            self.rope_pos_embeds_1d_audio
            # self.rope_pos_embeds, 
            # None
        )

        video_loss = F.mse_loss(video_pre, video_noise)
        audio_loss = F.mse_loss(audio_pre, audio_noise)
        total_loss = video_loss + audio_loss
        ## 4. check Loss Nan
        has_nan_loss = torch.isnan(total_loss).any().item()
        assert not has_nan_loss, "Loss has nan value!"
        ## 5. Log
        print(total_loss.item(), video_loss.item(), audio_loss.item())
        assert total_loss.dim() == 0
        del video_latents_noisy, video_noise, video_latents, video_prompt_embeds, audio_prompt_embeds, audio_latents_noisy, audio_noise, audio_latents, audio_duration_embeds, audio_rotary_embedding
        torch.cuda.empty_cache()
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_video", video_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss_audio", audio_loss, on_step=True, on_epoch=True, batch_size=batch_size)
        return total_loss


if __name__ == "__main__":
    pass
    

        
        
