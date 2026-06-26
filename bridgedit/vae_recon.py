from diffusers.models import AutoencoderKL
import torch
import numpy as np
from transformers import SpeechT5HifiGan
import librosa
import soundfile as sf
import os
from paths import model_path, BRIDGEDIT_ROOT

# 移除了 'mel_spectrogram_to_waveform' 函数，因为它被错误地定义和调用。
# 我们将在 audio_recon 中内联（inline）正确的逻辑。

def load_mel_spectrogram_from_audio(audio_path, sr=16000):
    """
    加载音频并将其转换为 VAE 期望的 4D mel spectrogram 张量。
    """
    y, _ = librosa.load(audio_path, sr=sr)
    
    # 使用关键字参数 y=y
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        fmin=0,
        fmax=sr // 2,
    )
    
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)
    
    # 关键修复：
    # 1. .unsqueeze(0) 添加 batch 维度 (B=1)
    # 2. .unsqueeze(1) 添加 channel 维度 (C=1)
    # 最终形状变为 [1, 1, 80, T] (B, C, H, W)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).unsqueeze(1)
    
    return mel_spectrogram

def audio_recon():
    audio = os.path.join(BRIDGEDIT_ROOT, 'dog2.wav')
    
    # 修复：使用 dtype 而不是 torch_dtype
    audio_vae = AutoencoderKL.from_pretrained(
        model_path("audioldm/vae"), local_files_only=True,
        ignore_mismatched_sizes=False,
        torch_dtype=torch.float16,  # <--- 修复
    )
    
    # 修复：使用 dtype 而不是 torch_dtype
    vocoder = SpeechT5HifiGan.from_pretrained(
        model_path("audioldm/vocoder"), local_files_only=True,
        ignore_mismatched_sizes=False,
        torch_dtype=torch.float16,  # <--- 修复
    )
    
    # 加载为 4D 张量 (B, C, H, W) -> [1, 1, 80, T]
    mel_spectrogram = load_mel_spectrogram_from_audio(audio)
    
    print(f"Mel Spectrogram 形状: {mel_spectrogram.shape}") # 应该输出 [1, 1, 80, ...]

    with torch.no_grad():
        # VAE 编码
        latent = audio_vae.encode(mel_spectrogram.half()).latent_dist.sample()
        print(f"Latent 形状: {latent.shape}")
        
        # VAE 解码
        # recon_mel 的形状将是 [1, 1, 80, T]
        recon_mel = audio_vae.decode(latent.half()).sample
        print(f"Recon Mel 形状: {recon_mel.shape}")

    # 修复：正确调用 Vocoder
    # 1. 准备 vocoder 的输入
    
    # HifiGan vocoder 期望 [B, H, T] (即 [1, 80, T])
    # 我们需要从 [1, 1, 80, T] 中移除 channel 维度
    if recon_mel.dim() == 4:
        recon_mel_for_vocoder = recon_mel.squeeze(1) # 形状变为 [1, 80, T]
    else:
        recon_mel_for_vocoder = recon_mel

    print(f"Vocoder 输入形状: {recon_mel_for_vocoder.shape}")

    # 2. 将 mel 转换为 vocoder 的 dtype (float16)
    recon_mel_for_vocoder = recon_mel_for_vocoder.to(vocoder.dtype)
    
    # 3. 调用 vocoder
    waveform = vocoder(recon_mel_for_vocoder)
    
    # 4. 转换为 CPU 和 float32 以进行保存
    waveform = waveform.cpu().float()

    # waveform 形状是 [1, T_audio]
    # audio_np[0] 形状是 [T_audio] (1D)
    audio_np = waveform[0].float().cpu().numpy()

    # 你的逻辑是正确的：将 1D 数组转换为 2D 立体声以便保存
    if audio_np.ndim == 1:
        # 从 (T_audio,) 变为 (T_audio, 1)
        audio_np = audio_np.reshape(-1, 1) 
        # 变为 (T_audio, 2)
        audio_np = np.repeat(audio_np, 2, axis=1)
        
    print(f"最终保存的音频形状: {audio_np.shape}")
    
    sf.write('recon.wav', audio_np, 16000)
    print("重建的音频已保存为 recon.wav")

if __name__ == "__main__":
    audio_recon()