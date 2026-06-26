import json
import os
import logging
from pathlib import Path
import torch
from einops import rearrange
import torchaudio
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from colorlog import ColoredFormatter

from av_bench.data.video_dataset import VideoDataset, error_avoidance_collate
from av_bench.synchformer.synchformer import Synchformer, make_class_grid
from av_bench.data.audio_dataset import SynchformerAudioDataset, pad_or_truncate
from av_bench.utils import (unroll_dict, unroll_dict_all_keys, unroll_paired_dict,
                            unroll_paired_dict_with_key)
import numpy as np
_syncformer_ckpt_path = os.path.join(os.environ.get("T2SV_MODEL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))), "synchformer/synchformer_state_dict.pth")

log = logging.getLogger()
device = 'cuda'

LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)

setup_eval_logging()

def encode_audio_with_sync(synchformer: Synchformer, x: torch.Tensor,
                           mel: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    with torch.no_grad():
        b, t = x.shape
        # partition the video
        segment_size = 10240
        step_size = 10240 // 2
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)
        x = mel(x)
        x = torch.log(x + 1e-6)
        x = pad_or_truncate(x, 66)

        mean = -4.2677393
        std = 4.5689974
        x = (x - mean) / (2 * std)
        # x: B * S * 128 * 66z
        x = synchformer.extract_afeats(x.unsqueeze(2))
    return x

def encode_video_with_sync(synchformer: Synchformer, x: torch.Tensor, segment_size: int, step_size: int) -> torch.Tensor:
    # x: (B, T, C, H, W) H/W: 224
    with torch.no_grad():
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224
        # partition the video
        num_segments = (t - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            segments.append(x[:, i * step_size:i * step_size + segment_size])
        
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)
        x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
        
        x = synchformer.extract_vfeats(x)
        x = rearrange(x, '(b s) 1 t d -> b s t d', b=b)
    return x




@torch.inference_mode()
def load_sychformer(path=_syncformer_ckpt_path):
    sync_model = Synchformer().to(device).eval()
    sd = torch.load(_syncformer_ckpt_path, weights_only=True)
    sync_model.load_state_dict(sd)
    return sync_model

def Cal_Desync(generated_video, generated_audio):
    logging.basicConfig(level=logging.INFO)
    task_name='t2v_v2a'
    batch_size=8
    cache_path='./cache'
    load_video_cache=False
    load_audio_cache=False
    audio_cache = os.path.join(cache_path, f'{task_name}_synchformer_audio.pth')
    video_cache = os.path.join(cache_path, f'{task_name}_synchformer_video.pth')
    os.makedirs(cache_path, exist_ok=True)
    # Load synchformer models and mel spectrogram
    log.info("Loading Synchformer models and Mel Spectrogram")
    sync_model = load_sychformer()
    sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    ).to(device)

    if load_video_cache:
        log.info('Extracting Video Cache...')
        sync_video_features = torch.load(video_cache, weights_only=True)
    else:
        # Extract Video Feature
        log.info('Extracting Video Feature')
        video_path = Path(generated_video)
        video_names = os.listdir(video_path)
        video_paths = [video_path / f for f in video_names if f.endswith('.mp4')]
        log.info(f'{len(video_paths)} videos found.')
        dataset = VideoDataset(video_paths, duration_sec=4.9)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=8,
                            collate_fn=error_avoidance_collate)
        output_sync_features = {}
        cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)
        # output_ib_features = {}
        for data in tqdm(loader):
            name = data['name']
            # ib_video = data['ib_video'].to(device)
            sync_video = data['sync_video'].to(device)
            sync_features = encode_video_with_sync(sync_model, sync_video, segment_size=16, step_size=2)
            sync_features = sync_features.cpu().detach()
            for i, n in enumerate(name):
                # saving a view will save the entire tensor so don't
                output_sync_features[n] = sync_features[i].clone()
        synchformer_video_feature_path = os.path.join(cache_path, f'{task_name}_synchformer_video.pth')
        log.info(f'Saving {len(output_sync_features)} features to {synchformer_video_feature_path}')
        sync_video_features = output_sync_features
        torch.save(output_sync_features, synchformer_video_feature_path)
    
    if load_audio_cache:
        log.info('Extracting Audio Cache...')
        sync_audio_features = torch.load(audio_cache, weights_only=True)
    else:
        # Extrac Audio Feature
        log.info('Extracting Audio Feature')
        audio_path = Path(generated_audio)
        audios = sorted(list(audio_path.glob('*.wav')) + list(audio_path.glob('*.flac')) + list(audio_path.glob('*.mp4')),
                        key=lambda x: x.stem)
        log.info(f'{len(audios)} audios found in {audio_path}')
        dataset = SynchformerAudioDataset(audios, duration=4.9)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.to(device)
            features = encode_audio_with_sync(sync_model, wav, sync_mel_spectrogram).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        synchformer_audio_feature_path = os.path.join(cache_path, f'{task_name}_synchformer_audio.pth')
        log.info(f'Saving {len(out_dict)} features to {synchformer_audio_feature_path}')
        sync_audio_features = out_dict
        torch.save(out_dict, synchformer_audio_feature_path)
    
    log.info("Start Evaluating DeSync Scores")
    paired_sync_video_features, paired_sync_audio_features = unroll_paired_dict(sync_video_features, sync_audio_features)
    
    total_samples = paired_sync_video_features.shape[0]
    total_sync_scores = []
    sync_grid = make_class_grid(-2, 2, 21)
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size)):
            sync_video_batch = paired_sync_video_features[i:i + batch_size].to(device)
            sync_audio_batch = paired_sync_audio_features[i:i + batch_size].to(device)
            logits = sync_model.compare_v_a(sync_video_batch[:, :14], sync_audio_batch[:, :14])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

            logits = sync_model.compare_v_a(sync_video_batch[:, -14:], sync_audio_batch[:, -14:])
            top_id = torch.argmax(logits, dim=-1).cpu().numpy()
            for j in range(sync_video_batch.shape[0]):
                total_sync_scores.append(abs(sync_grid[top_id[j]].item()))
        average_sync_score = np.mean(total_sync_scores)
        print("Average DeSync Score: ", average_sync_score)
@torch.inference_mode()
def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_dir", type=str, default='/mnt/task_runtime/t2av/result_video/t2v')
    parser.add_argument("--input_audio_dir", type=str, default='/mnt/task_runtime/t2av/result_audio/v2a')
    parser.add_argument("--task_name", type=str, default='t2v_v2a')
    parser.add_argument("--caption_file", type=str, default='/mnt/task_runtime/t2av/text_favd/test.jsonl')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_path", type=str, default='./cache')
    parser.add_argument("--load_video_cache", type=bool, default=False)
    parser.add_argument("--load_audio_cache", type=bool, default=False)
    args = parser.parse_args()
    audio_cache = os.path.join(args.cache_path, f'{args.task_name}_synchformer_audio.pth')
    video_cache = os.path.join(args.cache_path, f'{args.task_name}_synchformer_video.pth')
    batch_size = args.batch_size
    os.makedirs(args.cache_path, exist_ok=True)
    # Load synchformer models and mel spectrogram
    log.info("Loading Synchformer models and Mel Spectrogram")
    sync_model = load_sychformer()
    sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    ).to(device)

    if args.load_video_cache:
        log.info('Extracting Video Cache...')
        sync_video_features = torch.load(video_cache, weights_only=True)
    else:
        # Extract Video Feature
        log.info('Extracting Video Feature')
        video_path = Path(args.input_video_dir)
        video_names = os.listdir(video_path)
        video_paths = [video_path / f for f in video_names if f.endswith('.mp4')]
        log.info(f'{len(video_paths)} videos found.')
        dataset = VideoDataset(video_paths, duration_sec=4.9)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            collate_fn=error_avoidance_collate)
        output_sync_features = {}
        cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)
        # output_ib_features = {}
        for data in tqdm(loader):
            name = data['name']
            # ib_video = data['ib_video'].to(device)
            sync_video = data['sync_video'].to(device)
            sync_features = encode_video_with_sync(sync_model, sync_video, segment_size=16, step_size=2)
            sync_features = sync_features.cpu().detach()
            for i, n in enumerate(name):
                # saving a view will save the entire tensor so don't
                output_sync_features[n] = sync_features[i].clone()
        synchformer_video_feature_path = os.path.join(args.cache_path, f'{args.task_name}_synchformer_video.pth')
        log.info(f'Saving {len(output_sync_features)} features to {synchformer_video_feature_path}')
        sync_video_features = output_sync_features
        torch.save(output_sync_features, synchformer_video_feature_path)
    
    if args.load_audio_cache:
        log.info('Extracting Audio Cache...')
        sync_audio_features = torch.load(audio_cache, weights_only=True)
    else:
        # Extrac Audio Feature
        log.info('Extracting Audio Feature')
        audio_path = Path(args.input_audio_dir)
        audios = sorted(list(audio_path.glob('*.wav')) + list(audio_path.glob('*.flac')) + list(audio_path.glob('*.mp4')),
                        key=lambda x: x.stem)
        log.info(f'{len(audios)} audios found in {audio_path}')
        dataset = SynchformerAudioDataset(audios, duration=4.9)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)
        out_dict = {}
        for wav, filename in tqdm(loader):
            wav = wav.to(device)
            features = encode_audio_with_sync(sync_model, wav, sync_mel_spectrogram).cpu()
            for i, f_name in enumerate(filename):
                out_dict[f_name] = features[i]
        synchformer_audio_feature_path = os.path.join(args.cache_path, f'{args.task_name}_synchformer_audio.pth')
        log.info(f'Saving {len(out_dict)} features to {synchformer_audio_feature_path}')
        sync_audio_features = out_dict
        torch.save(out_dict, synchformer_audio_feature_path)
    
    log.info("Start Evaluating DeSync Scores")
    paired_sync_video_features, paired_sync_audio_features = unroll_paired_dict(sync_video_features, sync_audio_features)
    
    total_samples = paired_sync_video_features.shape[0]
    total_sync_scores = []
    sync_grid = make_class_grid(-2, 2, 21)
    for i in tqdm(range(0, total_samples, batch_size)):
        sync_video_batch = paired_sync_video_features[i:i + batch_size].to(device)
        sync_audio_batch = paired_sync_audio_features[i:i + batch_size].to(device)
        logits = sync_model.compare_v_a(sync_video_batch[:, :14], sync_audio_batch[:, :14])
        top_id = torch.argmax(logits, dim=-1).cpu().numpy()
        for j in range(sync_video_batch.shape[0]):
            total_sync_scores.append(abs(sync_grid[top_id[j]].item()))

        logits = sync_model.compare_v_a(sync_video_batch[:, -14:], sync_audio_batch[:, -14:])
        top_id = torch.argmax(logits, dim=-1).cpu().numpy()
        for j in range(sync_video_batch.shape[0]):
            total_sync_scores.append(abs(sync_grid[top_id[j]].item()))
    average_sync_score = np.mean(total_sync_scores)
    print("Average DeSync Score: ", average_sync_score)

if __name__ == '__main__':
    
    main()
