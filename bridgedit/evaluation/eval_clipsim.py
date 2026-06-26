import clip
import torch
import numpy as np
import cv2
import os
import argparse
import json
from tqdm import tqdm
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
def normalize(data):
    return (data/255.0-v_mean)/v_std

def frames2tensor(vid_list, fnum=3, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube

class Clip_Score:   
    def __init__(self, path='/fs/fast/share/aimind_files/video_eval/models/ViT-B-16/ViT-B-16.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(path, self.device)
        self.clip_model.eval()  # 设置为评估模式
    
    def calculate_clip_score(self, frames_tensor, text):
        frames = frames_tensor.squeeze(0)
        batch_size = 4  # 调整批次大小以适应显存
        features = []
        with torch.no_grad():  # 禁用梯度
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                batch_features = self.clip_model.encode_image(batch)
                features.append(batch_features)
            video_feature = torch.cat(features, dim=0).mean(dim=0)
            video_feature /= video_feature.norm()
            text_feature = self.clip_model.encode_text(clip.tokenize([text]).to(self.device))
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            return (video_feature @ text_feature.T).item()

def process_video_caption_test(args):
    """
    Input:
        input_video_dir: dictory of mp4 files
        caption_file: json
    Output:
        list of json [{video_name:video_caption}]
    """
    print("Processing video paths")
    # video_dir = os.listdir(args.input_video_dir)
    # with open(args.caption_file, 'r') as f:
    #     caption_js_list = [json.loads(line) for line in f.readlines()]
    # video2caption = []
    # for caption_js in caption_js_list:
    #     js_new = {'path': os.path.join(args.input_video_dir,caption_js['id'])+'.mp4','caption':caption_js['caption']}
    #     video2caption.append(js_new)
    with open(args.caption_file, 'r') as f:
        caption_js = json.load(f)
    video2caption = []
    for video_path in caption_js:
        video_id = os.path.basename(video_path)
        js_new = {'path': os.join, 'caption': caption_js[video_path]['video_caption']}
        video2caption.append(js_new)
    return video2caption

def process_video_caption(generated_video, seperate, caption_file):
    """
    Input:
        input_video_dir: dictory of mp4 files
        caption_file: json
    Output:
        list of json [{video_name:video_caption}]
    """
    print("Processing video paths")
    video_dir = os.listdir(generated_video)
    # with open(caption_file, 'r') as f:
    #     caption_js_list = [json.loads(line) for line in f.readlines()]
    with open(caption_file, 'r') as f:
        caption_js = json.load(f)
    video2caption = []
    for video_path in caption_js:
        video_id = os.path.basename(video_path)
        js_new = {'path': os.path.join(generated_video, video_id), 'caption': caption_js[video_path]['video_caption']}
        video2caption.append(js_new)
    return video2caption

def calculate_clip_sim(clip_model, video2caption, device= "cuda" if torch.cuda.is_available() else "cpu"):
    clip_model.eval()  # 切换到推理模式
    clipscores = []
    for video_js in tqdm(video2caption):
        video = cv2.VideoCapture(video_js['path'])
        frames = [x for x in _frame_from_video(video)]
        try:
            frames_tensor = frames2tensor(frames, device=device)
            frames_tensor = frames_tensor.squeeze(0)
        except:
            continue
        caption = video_js['caption']
        with torch.no_grad():
            video_feature = clip_model.encode_image(frames_tensor).mean(dim=0)
            video_feature /= video_feature.norm()

        max_length = 50
        # Split the caption into chunks
        words = caption.split(' ')
        chunks = [words[i:i+max_length] for i in range(0, len(words), max_length)]
        # Process each chunk
        text_features = []
        with torch.no_grad():
            for chunk in chunks:
                chunk_caption = ' '.join(chunk)
                tokens = clip.tokenize([chunk_caption]).to(device)
                text_feature = clip_model.encode_text(tokens)
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
                text_features.append(text_feature)
                del text_feature
        # Combine features if needed
        final_text_feature = torch.cat(text_features, dim=0)
        # text_feature = clip_model.encode_text(clip.tokenize([caption]).to(device))
        # text_feature /= text_feature.norm(dim=-1, keepdim=True)
        clip_score = video_feature @ final_text_feature.T
        clip_score = torch.max(clip_score)
        del frames_tensor, video_feature, final_text_feature, text_features
        torch.cuda.empty_cache()
        clipscores.append(clip_score.detach().cpu().numpy())
        del clip_score
    return clipscores


def Cal_Clipsim(generated_video, seperate, caption_file):
    path = os.path.join(os.environ.get("T2SV_MODEL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))), "clip/ViT-L-14.pt")
    video2caption = process_video_caption(generated_video, seperate, caption_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(path, device)
    clipscores = calculate_clip_sim(clip_model, video2caption)
    print(f"CLIPSIM: {sum(clipscores)/len(clipscores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_dir", type=str, default='/mnt/task_runtime/dataset/favdbench/video/test', help='Insert the videos folder path')
    parser.add_argument("--caption_file", type=str, default='/mnt/task_runtime/t2av/text_favd/test.jsonl')
    # parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    path = os.path.join(os.environ.get("T2SV_MODEL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))), "clip/ViT-L-14.pt")
    video2caption = process_video_caption_test(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load(path, device)
    clipscores = calculate_clip_sim(args, clip_model, video2caption)
    print(f"CLIPSIM: {sum(clipscores)/len(clipscores):.4f}")
    # video = cv2.VideoCapture('/mnt/task_runtime/t2av/result_video/output2.mp4')
    # frames = [x for x in _frame_from_video(video)]
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # frames_tensor = frames2tensor(frames, device=device)
    # frames_tensor = frames_tensor.squeeze(0)
    # # print(frames_tensor)
    # path = os.path.join(os.environ.get("T2SV_MODEL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))), "clip/ViT-B-16.pt")
    # clip_model, _ = clip.load(path, device)
    # text = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
    # video_feature = clip_model.encode_image(frames_tensor).mean(dim=0)
    # video_feature /= video_feature.norm()
    # text_feature = clip_model.encode_text(clip.tokenize([text]).to(device))
    # text_feature /= text_feature.norm(dim=-1, keepdim=True) 
    # # print(text_feature.shape)
    # # print(clip_model)
    # clip_score = video_feature @ text_feature.T
    # print(clip_score.detach().cpu().numpy())