import os, csv
import numpy as np
import torch
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

def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[200, 360], antialias=False)
    img2_batch = F.resize(img2_batch, size=[200, 360], antialias=False)
    return transforms(img1_batch, img2_batch)

def calculate_motion_score(video_path):
    frames, _, _ = read_video(str(video_path), output_format="TCHW", pts_unit="sec")
    
    total_flow = 0
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
            flow_magnitude = torch.mean(torch.abs(predicted_flows))
            total_flow += flow_magnitude.item() * min_len
    
    # return total_flow / (len(frames) - 1)
    return total_flow 

def process_video_folder(folder_path):
    results = {}
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # 使用tqdm创建进度条
    for filename in tqdm(video_files, desc="Processing videos", unit="video"):
        video_path = os.path.join(folder_path, filename)
        try:
            motion_score = calculate_motion_score(video_path)
            results[filename] = motion_score
            # tqdm.write(f"Processed {filename}: Motion Score = {motion_score}")
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")
    
    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        for video, score in results.items():
            f.write(f"{video}: {score}\n")

# 主程序
if __name__ == "__main__":
    
    # base_dir = "/root/workspace/bridiff_i2va/result"
    base_dir = "/root/workspace/bridiff_i2va/data"

    # 首先，从CSV文件中读取已有的数据
    existing_scores = {}
    # with open('./ms_ablation.csv', 'r') as csv_file:
    #     reader = csv.reader(csv_file)
    #     for row in reader:
    #         tag, score = row
    #         existing_scores[tag] = score

    # 打开CSV文件，准备写入结果
    with open('./ms_cog.csv', 'a') as csv_file:
        writer = csv.writer(csv_file)

        # 遍历base_dir下的所有子目录
        for dir_name in os.listdir(base_dir):
            dir_path = os.path.join(base_dir, dir_name)

            
            try:
                # 如果这是一个目录，并且它的名字中包含"multiguidance"
                if os.path.isdir(dir_path) and "repeat" in dir_name:
                    # 从目录名中提取tag
                    # tags = dir_name.split("_skipdit_")[1].split("_", 2)[:2]
                    # tag = "_".join(tags)
                    tag = dir_name

                    if tag in existing_scores:
                        continue

                    # video_dir = os.path.join(dir_path, "predict/video")
                    video_dir = dir_path
                    
                    results = process_video_folder(video_dir)
                    ms = sum(results.values()) / len(results)
                    
                    print(f"MS {tag} = {ms}")

                    # 将结果写入CSV文件
                    writer.writerow([tag, ms,])
            except:
                print(dir_name)
                
                
    # video_folder = "/root/workspace/bridiff_i2va/result/avsync15_va_dualdit_e899"  # 替换为您的视频文件夹路径
    # output_file = "motion_scores.txt"  # 结果输出文件
    
    # results = process_video_folder(video_folder)
    # save_results(results, output_file)
    # print(f"Results saved to {output_file}")
