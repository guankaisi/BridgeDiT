"""
AV-Align Metric: Audio-Video Alignment Evaluation

AV-Align is a metric for evaluating the alignment between audio and video modalities in multimedia data.
It assesses synchronization by detecting audio and video peaks and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.

Usage:
- Provide a folder of video files as input.
- The script calculates the AV-Align score for the set of videos.
"""


import argparse, glob, cv2, json, os, pdb, random, csv
import os.path as osp
import numpy as np
import librosa
import librosa.display
from sympy import N
from dtaidistance import dtw
import torchaudio, torch
# from torchaudio.transforms import DownmixMono

import torchvision.transforms.functional as F
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from tqdm import tqdm


# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和权重
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()
model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

# cache_path = "./video_cache.json"
cache_json = None

# resie frames
def resize_frames(frames, new_size_scheme):
    """
    Args:
        frames (list): the elements in frames are numpy.ndarray.
        new_size_scheme (str):  resize scheme.
    Return:
        frames: the elements in list are resized frames.
    """
    h, w, _ = frames[0].shape
    # new_w, new_h = w, h
    if new_size_scheme.startswith("min"):
        min_edge = int(new_size_scheme.split("=")[1])
        scale_ratio = min_edge / min(w, h)
        new_h = int(scale_ratio * h)
        new_w = int(scale_ratio * w)
    elif new_size_scheme.find(":") != -1:
        new_w = int(new_size_scheme.split(":")[0])
        new_h = int(new_size_scheme.split(":")[1])

    if (w, h) == (new_w, new_h):
        return frames

    new_frames = []
    for img in frames:
        new_frames.append(
            cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR))

    return new_frames

# Function to extract frames from a video file
def extract_frames(video_path, resize_scheme=None, frame_num=14):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        frames (list): List of frames extracted from the video.
        frame_rate (float): Frame rate of the video.
    """

    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open the video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if resize_scheme is not None:
        frames = resize_frames(frames, resize_scheme)

    if len(frames) < frame_num:
        # repeat the last frame
        if len(frames) / frame_rate < 0.7:
            raise ValueError("Error: The video is too short.")
            # repeat the last frame
        frames += [frames[-1]] * (frame_num - len(frames))
    # elif len(frames) > frame_num:
    #     if len(frames) / frame_rate > 1.4:
    #         raise ValueError("Error: The video is too long.")
    #     frames = frames[:frame_num]
    return frames, frame_rate

# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        onset_times (list): List of times (in seconds) where audio peaks occur.
    """
    # y, sr = librosa.load(audio_file)
    waveform, sr = torchaudio.load(audio_file)
    # if waveform.shape[0] == 2:
    #     # Create a transformer to downmix audio to mono
    #     # transformer = DownmixMono()
    #     y = waveform[0]
    # else:
    #     y = waveform
    y = waveform[0].numpy()
    
    # # Calculate the onset envelope
    # onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # onset_env = torch.tensor(onset_env)
    # # Min Max Normalization
    # onset_env = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min())
    # # Mean Standardization
    # # onset_env = (onset_env - onset_env.mean()) / onset_env.std()
    # # onset_env = (onset_env - .0) / (onset_env.max() - .0)
    # # Set to zero if the value is less than 0.5
    # onset_env[onset_env < 0.3] = 0
    
    # return onset_env


    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times

# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    video_peaks = find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks

# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude

# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_dtw(audio_peaks, video_peaks,):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    # print(f'audio_peaks:{audio_peaks},')
    # print(f'video_peaks:{video_peaks}')
                
    return dtw.distance(audio_peaks, video_peaks)

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[200, 360], antialias=False)
    img2_batch = F.resize(img2_batch, size=[200, 360], antialias=False)
    return transforms(img1_batch, img2_batch)

def calculate_motion_score(video_path):
    frames, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
    
    total_flow = torch.tensor([]).to(device)
    batch_size = 32  # 可以根据你的GPU内存调整这个值
    
    with torch.no_grad():
        for i in range(0, len(frames) - 1, batch_size):
            img1_batch = frames[i:i+batch_size]
            img2_batch = frames[i+1:i+batch_size+1]
            
            # 确保两个batch的大小相同
            min_len = min(len(img1_batch), len(img2_batch))
            img1_batch = img1_batch[:min_len]
            img2_batch = img2_batch[:min_len]
            
            # 预处理批次
            img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
            
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            predicted_flows = list_of_flows[-1]
            flow_magnitude = torch.mean(torch.abs(predicted_flows), dim=[1,2,3])
            total_flow = torch.cat([total_flow, flow_magnitude])
    
    video_peaks = total_flow
    # print(video_peaks)
    # Min Max Normalization
    video_peaks = (video_peaks - video_peaks.min()) / (video_peaks.max() - video_peaks.min())
    # Mean Standardization
    # video_peaks = (video_peaks - video_peaks.mean()) / video_peaks.std()
    # video_peaks = (video_peaks - .0) / (video_peaks.max() - .0)
    # Set to zero if the value is less than 0.5
    
    # video_peaks[video_peaks < 0.3] = 0
    
    return video_peaks.cpu()

# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """

    local_extrema_indexes = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:  # Local maximum
            local_extrema_indexes.append(i / fps)

    return local_extrema_indexes


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    # print(f'audio_peaks:{audio_peaks}, video_peaks:{video_peaks}')
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                used_video_peaks[j] = True
                break
    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length)

def Cal_Avalign(generated_video, generated_audio, ends='.mp4'):
    files = [file for file in glob.glob(osp.join(generated_video, '*.mp4'))]
    random.seed(0)
    random.shuffle(files)
    score = 0
    # files = files[:30]
    index=0
    error_count = 0
    for file in tqdm(files, desc="Process Videos"):
        file = file[:-4]
        video_path = f'{file}.mp4'
        video_name = osp.basename(video_path)[:-4]
        if ends == '.mp4':
            audio_path = f'{generated_audio}/{video_name}.mp4'
        else:
            audio_path = f'{generated_audio}/{video_name}.wav'
        if not os.path.exists(audio_path):
            print(f"The audio path '{audio_path}' not exists.")
            continue
        
        try:
            audio_peaks = detect_audio_peaks(audio_path)
            # video_frames, fps = extract_frames(video_path)
            # flow_trajectory, video_peaks = detect_video_peaks(video_frames, fps=fps)
            video_peaks = calculate_motion_score(video_path)
            

            tmp_score=calc_dtw(audio_peaks, video_peaks,)
            # tmp_score = calc_intersection_over_union(audio_peaks, video_peaks, fps=fps)
            score += tmp_score
            # print(f'index:{index}, score:{tmp_score}')
            index+=1
        except Exception as e:
            error_count += 1
            print("error_count: ", error_count)
            continue

    print('AV-Align: ', score/(len(files)-error_count)) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_dir", type=str, default='/mnt/task_runtime/t2av/result_video/t2v', help='Insert the videos folder path')
    parser.add_argument("--input_wav_dir", type=str, default='/mnt/task_runtime/t2av/result_wav/t2a', help='Insert the videos folder path')
    parser.add_argument("--size", type=str, default=None, help="resize frames",)
    parser.add_argument("--cache_path", type=str, default='./cache.json', help="resize frames",)
    args = parser.parse_args()

    files = [file for file in glob.glob(osp.join(args.input_video_dir, '*.mp4'))]
    score = 0
    cache_path = args.cache_path
    if cache_json is None and osp.exists(cache_path):
        cache_json = json.load(open(cache_path, "r"))
    else:
        cache_json = {}

    random.seed(0)
    random.shuffle(files)
    # files = files[:30]
    
    index=0
    for file in tqdm(files, desc="Process Videos"):

        file = file[:-4]
        video_path = f'{file}.mp4'
        video_name = osp.basename(video_path)[:-4]
        # audio_path = f'{args.input_wav_dir}/{video_name}.mp4'
        audio_path = f'{args.input_wav_dir}/{video_name}.wav'
        if not os.path.exists(audio_path):
            print(f"The audio path '{audio_path}' not exists.")
            continue

        audio_peaks = detect_audio_peaks(audio_path)
        # video_peaks = detect_video_peaks(video_path, fps=10)
        video_peaks = calculate_motion_score(video_path)
        

        tmp_score=calc_dtw(audio_peaks, video_peaks,)
        # tmp_score = calc_intersection_over_union(audio_peaks, video_peaks, fps=10)
        score += tmp_score
        # print(f'index:{index}, score:{tmp_score}')
        index+=1

    print('AV-Align: ', score/len(files)) 
    
    
    
    """ Multigaidence """
    
    # base_dir = "/root/workspace/bridiff_i2va/log"

    # # 首先，从CSV文件中读取已有的数据
    # existing_scores = {}
    # if os.path.exists('./avalign_ablation.csv'):
    #     with open('./avalign_ablation.csv', 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         for row in reader:
    #             tag, score = row
    #             existing_scores[tag] = score

    # # 打开CSV文件，准备写入结果
    # with open('./avalign_ablation.csv', 'a') as csv_file:
    #     writer = csv.writer(csv_file)

    #     # 遍历base_dir下的所有子目录
    #     for dir_name in os.listdir(base_dir):
    #         try:
    #             dir_path = os.path.join(base_dir, dir_name)

    #             # 如果这是一个目录，并且它的名字中包含"multiguidance"
    #             if os.path.isdir(dir_path) and "e39_infer_on_avsync15_mgsubset" in dir_name:

    #                 # 从目录名中提取tag
    #                 # tags = dir_name.split("_skipdit_")[1].split("_", 2)[:2]
    #                 tags = dir_name
    #                 tag = "_".join(tags)

    #                 if tag in existing_scores:
    #                     continue

    #                 video_dir = os.path.join(dir_path, "predict/video")
                    
    #                 # print(video_dir)
    #                 files = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if file.endswith(".mp4")]
    #                 score = 0
                    
    #                 index=0
    #                 for file in tqdm(files, desc="Process Videos"):

    #                     audio_peaks = detect_audio_peaks(file)
    #                     video_peaks = calculate_motion_score(file)
                        
    #                     tmp_score=calc_dtw(audio_peaks, video_peaks,)
    #                     score += tmp_score
    #                     # print(f'index:{index}, score:{tmp_score}')
    #                     index+=1

    #                 print(f'AV-Align {tag}: ', score/len(files)) 
                    
    #                 # 将结果写入CSV文件
    #                 writer.writerow([tag, score/len(files)])
    #         except BaseException as e:
    #             print(e)
    #             print(dir_name)
    
    
    # """ VA """
    
    # # base_dir = "/root/workspace/bridiff_i2va/result"
    # base_dir = "/root/workspace/bridiff_i2va/data"

    # # 首先，从CSV文件中读取已有的数据
    # existing_scores = {}
    # if os.path.exists('./avalign_score_va_cog.csv'):
    #     with open('./avalign_score_va_cog.csv', 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         for row in reader:
    #             tag, score = row
    #             existing_scores[tag] = score


    # # 遍历base_dir下的所有子目录
    # for dir_name in os.listdir(base_dir):
    #     dir_path = os.path.join(base_dir, dir_name)

    #     # 如果这是一个目录，并且它的名字中包含"multiguidance"
    #     if os.path.isdir(dir_path) and ("repeat5" in dir_name ):
    #         tag = dir_name

    #         if tag in existing_scores:
    #             continue

    #         video_dir = dir_path
            
    #         files = [file for file in glob.glob(osp.join(video_dir, '*.mp4'))]
    #         score = 0
            
    #         index=0
    #         for file in tqdm(files, desc="Process Videos"):

    #             audio_peaks = detect_audio_peaks(file)
    #             video_peaks = calculate_motion_score(file)
    #             video_peaks = find_local_max_indexes(video_peaks, 7)
                
    #             tmp_score = calc_intersection_over_union(audio_peaks, video_peaks, 7)
    #             # tmp_score=calc_dtw(audio_peaks, video_peaks,)
                
    #             score += tmp_score
    #             # print(f'index:{index}, score:{tmp_score}')
    #             index+=1

    #         print(f'AV-Align {tag}: ', score/len(files)) 
            
    #         with open('./avalign_score_va_cog.csv', 'a') as csv_file:
    #             writer = csv.writer(csv_file)
    #             writer.writerow([tag, score/len(files)])
        
