import os, sys, wandb
from datetime import datetime, timezone, timedelta
import torch
import pytorch_lightning as pl
import json
import torch.multiprocessing as mp
from tqdm import tqdm
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from sana_audioldm_joint import Seperate_infer
from dataset.dataset_va import VGGSoundDataset, AvsyncDataset, VGGSoundSSDataset, LandscapeDataset
from models.audioldm.pipeline_audioldm import AudioLDMPipeline
from paths import REPO_ROOT, BRIDGEDIT_ROOT, caption_path, model_path
def predict():
    video_prompt = "In a serene backyard setting, a German Shepherd stands alert on a stone-paved path beside a crystal-clear swimming pool, the water shimmering under the soft, diffused sunlight. The atmosphere is tranquil, with lush greenery flanking the path and a white lounge chair resting invitingly near the pool's edge. The dog, with its sleek black and tan coat, pricks its ears forward, its body tense with anticipation. Its eyes focus intently on something off-screen, and its tail wags slightly, creating a subtle breeze that rustles the nearby foliage. As the dog begins to move, its paws press firmly against the cool stone tiles, sending tiny pebbles skittering aside. It leaps forward with a burst of energy, its muscles contracting powerfully, propelling it through the air. The dog's landing is graceful yet forceful, its paws striking the ground with enough impact to send a faint tremor through the earth. The surrounding plants sway gently in response to the dog's swift movements, their leaves brushing softly against each other. The scene is captured in a medium shot, the camera remaining steady, allowing the viewer to fully absorb the dynamic interplay between the dog's actions and the peaceful garden environment. The visual style is naturalistic and highly detailed, with soft, natural lighting that enhances the textures of the dog's fur, the stone path, and the vibrant greenery, all rendered in crisp, 8K quality."
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_prompt = "The gentle rustle of leaves and the soft swaying of plants accompany the alert barks of a German Shepherd, creating a tranquil yet energetic soundscape."
    audio_negative_prompts = "Low quality, unclear and noisy."
    model = Seperate_infer(config_path="config/sample_sana.yaml")
    model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts,save_name="test114")

def predict_audioldm():
    import torch

    repo_id = model_path("audioldm")
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    prompt = "A dog is barking."
    audio = pipe(prompt, num_inference_steps=50, audio_length_in_s=5.4).audios[0]
    import scipy
    scipy.io.wavfile.write("dog.wav", rate=16000, data=audio)

def process_subset(subset_keys, gpu_index):
    prompt_file = caption_path("recaption/avsync-test-72B-captions.json")
    prompt_json = json.load(open(prompt_file, 'r'))
    video_negative_prompts = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    audio_negative_prompts = "Low quality, unclear and noisy."
    
    torch.cuda.set_device(gpu_index)
    model = Seperate_infer(config_path="config/sample_sana.yaml")
    for key in subset_keys:
        video_name = os.path.basename(key)
        if video_name in os.listdir(os.path.join(BRIDGEDIT_ROOT, 'save_videos/seperate')):
            continue
        video_id = os.path.basename(key)[:-4]
        video_prompt = prompt_json[key]['video_caption']
        # audio_prompt = prompt_json[key]['audio_caption']
        audio_prompt = video_prompt
        model.predict_step(video_prompt, audio_prompt, video_negative_prompts, audio_negative_prompts, save_name=video_id)

def predict_avsync_parallel():
    num_gpus = 6  # 假设有 8 张 GPU
    prompt_file = caption_path("recaption/avsync-test-72B-captions.json")
    prompt_json = json.load(open(prompt_file, 'r'))
    keys = list(prompt_json.keys())
    num_keys = len(keys)
    
    # 将 keys 分成 num_gpus 份
    subset_sizes = [num_keys // num_gpus + (1 if i < num_keys % num_gpus else 0) for i in range(num_gpus)]
    subsets = []
    start = 0
    for size in subset_sizes:
        subsets.append(keys[start:start + size])
        start += size
    
    # 创建并启动 8 个进程
    processes = []
    for i in range(num_gpus):
        p = mp.Process(target=process_subset, args=(subsets[i], i))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()

if __name__ == "__main__":
    # predict_audioldm()
    # predict()
    predict_avsync_parallel()