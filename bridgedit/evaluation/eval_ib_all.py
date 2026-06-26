import os, json, csv, re, glob
import numpy as np
from tqdm import tqdm
import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# 在你的脚本最开始的地方加上这句！！！
torch.autograd.set_detect_anomaly(True)
def process_videos_test(video_paths, audio_paths, captions, model, device, batch_size):
    """Process videos and calculate multiple modality similarity scores."""
    ib_scores_va = []
    ib_scores_vt, ib_scores_at = [], []  # For text if needed
    for i in tqdm(range(0, len(video_paths), batch_size)):
        batch_video = video_paths[i:i+batch_size]
        batch_audio = audio_paths[i:i+batch_size]
        batch_captions = captions[i:i+batch_size] 
        try:
            # Load video+audio (+text) embeddings
            inputs = {
                ModalityType.VISION: data.load_and_transform_video_data(batch_video, device),
                ModalityType.AUDIO: data.load_and_transform_audio_data(batch_audio, device),
            }
            
            inputs[ModalityType.TEXT] = data.load_and_transform_text(batch_captions, device)
            
            with torch.no_grad():
                embeddings = model(inputs)
                embeddings_v = embeddings[ModalityType.VISION]
                embeddings_a = embeddings[ModalityType.AUDIO]
                embeddings_t = embeddings[ModalityType.TEXT]
            # Normalize embeddings
            embeddings_v = torch.nn.functional.normalize(embeddings_v, dim=-1)
            embeddings_a = torch.nn.functional.normalize(embeddings_a, dim=-1)
            # Calculate similarity scores
            va_scores = (embeddings_v * embeddings_a).sum(dim=1).mean().item()
            ib_scores_va.append(va_scores)    
            embeddings_t = torch.nn.functional.normalize(embeddings_t, dim=-1)
            vt = (embeddings_v * embeddings_t).sum(dim=1).mean().item()
            at = (embeddings_a * embeddings_t).sum(dim=1).mean().item()
            ib_scores_vt.append(vt)
            ib_scores_at.append(at)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
    # Compute averages
    if not ib_scores_va:
        raise RuntimeError("All ImageBind batches failed; no valid scores were produced.")
    avg_va = np.mean(ib_scores_va)
    avg_vt = np.mean(ib_scores_vt)
    avg_at = np.mean(ib_scores_at)
    return avg_va, avg_vt, avg_at

def process_videos(video_paths, audio_paths, video_captions, audio_captions, model):
    """Process videos and calculate multiple modality similarity scores."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    ib_scores_va = []
    ib_scores_vt, ib_scores_at = [], []  # For text if needed
    for i in tqdm(range(0, len(video_paths), batch_size)):
        try:
            batch_video = video_paths[i:i+batch_size]
            batch_audio = audio_paths[i:i+batch_size]
            batch_video_captions = video_captions[i:i+batch_size] 
            batch_audio_captions = audio_captions[i:i+batch_size] 
            with torch.no_grad():
                vision_data = data.load_and_transform_video_data(batch_video, device)
                audio_data = data.load_and_transform_audio_data(batch_audio, device)
                text_data = data.load_and_transform_text(batch_video_captions, device)
                inputs = {
                    ModalityType.VISION: vision_data,
                    ModalityType.AUDIO: audio_data,
                    ModalityType.TEXT: text_data,
                }
                embeddings = model(inputs)
                embeddings_v = embeddings[ModalityType.VISION]
                embeddings_a = embeddings[ModalityType.AUDIO]
                embeddings_vt = embeddings[ModalityType.TEXT]
                inputs = {ModalityType.TEXT: data.load_and_transform_text(batch_audio_captions, device)}
                embeddings = model(inputs)
                embeddings_at = embeddings[ModalityType.TEXT]
                # Normalize embeddings
                embeddings_v = torch.nn.functional.normalize(embeddings_v, dim=-1)
                embeddings_a = torch.nn.functional.normalize(embeddings_a, dim=-1) 
                embeddings_vt = torch.nn.functional.normalize(embeddings_vt, dim=-1)
                embeddings_at = torch.nn.functional.normalize(embeddings_at, dim=-1)
                va_scores = (embeddings_v * embeddings_a).sum(dim=1).mean().item()
                
                # 检查是否存在 NaN 或 inf
                vt_scores = (embeddings_v * embeddings_vt).sum(dim=1).mean().item()
                at_scores = (embeddings_a * embeddings_at).sum(dim=1).mean().item()
                ib_scores_va.append(va_scores)  
                ib_scores_vt.append(vt_scores)
                ib_scores_at.append(at_scores)

        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue


    # Compute averages
    if not ib_scores_va:
        raise RuntimeError("All ImageBind batches failed; no valid scores were produced.")
    avg_va = np.mean(ib_scores_va)
    avg_vt = np.mean(ib_scores_vt)
    avg_at = np.mean(ib_scores_at)
    return avg_va, avg_vt, avg_at

def setup_model(device):
    model = imagebind_model.imagebind_huge(pretrained=True)
    return model.eval().to(device)


def get_and_check_files(video_dir, audio_dir):
    def sorted_files(directory):
        return sorted([os.path.join(root, f) 
                      for root, _, files in os.walk(directory) 
                      for f in files])
    
    video_paths = sorted_files(video_dir)
    audio_paths = sorted_files(audio_dir)
    
    # Validate filename correspondence
    video_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]
    audio_names = [os.path.splitext(os.path.basename(p))[0] for p in audio_paths]
    # assert video_names == audio_names, "Filename mismatch between video/audio"
    return video_paths, audio_paths


def ib_va_test(video_dir, audio_dir, caption_file, batch_size, device):
    model = setup_model(device)
    video_paths, audio_paths = get_and_check_files(video_dir, audio_dir)
    with open(caption_file, 'r') as f:
        json_list = [json.loads(line) for line in f.readlines()]
    # align video_paths with caption
    captions = []
    for video in video_paths:
        video_id = video.split('/')[-1][:-4]
        for js in json_list:
            if video_id == js['id']:
                captions.append(js['caption'])
    return process_videos(video_paths, audio_paths, captions, model, device, batch_size)[:3]  # Return (va, vt, at)



def save_results(results_dict, output_dir):
    """保存评测结果到CSV文件"""
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    
    # 如果文件不存在则写入表头
    write_header = not os.path.exists(csv_path)
    
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = list(results_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            writer.writerow(results_dict)
            
    except Exception as e:
        print(f"保存结果失败: {str(e)}")
        # 尝试保存到备用路径
        try:
            alt_path = os.path.expanduser("~/eval_results_backup.csv")
            with open(alt_path, 'a') as f:
                f.write(str(results_dict) + "\n")
            print(f"结果已备份到: {alt_path}")
        except:
            print("无法保存结果，请检查磁盘空间和权限")

def Cal_Ib(generated_video, generated_audio, target_video_path, video_caption_path, audio_caption_path, seperate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = setup_model(device)
    # video_paths, audio_paths = get_and_check_files(generated_video, generated_audio)
    # print(video_paths)
    # print(audio_paths)
    video_paths, audio_paths = [], []
    video_captions, audio_captions = [], []
    with open(video_caption_path, 'r') as f:
        video_json = json.load(f)
    # for video_path in video_json:
    #     video_paths.append(video_path)
    #     audio_paths.append(video_path)
    for video_file in os.listdir(generated_video):
        if not video_file.endswith('.mp4'):
            continue
        video_paths.append(os.path.join(generated_video, video_file))
        audio_paths.append(os.path.join(generated_audio, video_file))
        # print(video_json.keys())
        video_captions.append(video_json[os.path.join(target_video_path, video_file)]['video_caption'])
        if seperate:
            audio_captions.append(video_json[os.path.join(target_video_path, video_file)]['audio_caption'])
        else:
            audio_captions.append(video_json[os.path.join(target_video_path, video_file)]['video_caption'])
    # with open(video_caption_path, 'r') as f:
    #     video_json_list = [json.loads(line) for line in f.readlines()]
    
    # with open(audio_caption_path, 'r') as f:
    #     audio_json_list = [json.loads(line) for line in f.readlines()]
    # align video_paths with caption
    # video_captions, audio_captions = [], []
    # for video in video_paths:
    #     video_id = video.split('/')[-1][:-4]
    #     for js in video_json_list:
    #         if video_id == js['id']:
    #             if seperate:
    #                 video_captions.append(js['video_caption'])
            
    # for audio in audio_paths:
    #     audio_id = audio.split('/')[-1][:-4]
    #     for js in audio_json_list:
    #         if audio_id == js['id']:
    #             if seperate:
    #                 audio_captions.append(js['audio_caption'])
    #             else:
    #                 audio_captions.append(js['caption'])
    va, vt, at = process_videos(video_paths, audio_paths, video_captions, audio_captions, model)
    print(f"VA: {va:.4f}, VT: {vt:.4f}, AT: {at:.4f}")

if __name__=="__main__":
    # # Example usage
    video_dir = "/mnt/task_runtime/dataset/avsync15/test"
    audio_dir = "/mnt/task_runtime/dataset/avsync15/test"
    video_caption_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/avsync_test.json'
    audio_caption_file = '/mnt/task_runtime/t2av/caption_pipeline/recaption/avsync_test.json'
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Cal_Ib(video_dir, audio_dir, video_caption_file, audio_caption_file, seperate=True)
    # va, vt, at = ib_va_test(video_dir, audio_dir, caption_file, batch_size, device)
    # print(f"VA: {va:.4f}, VT: {vt:.4f}, AT: {at:.4f}")
     
    

    # def find_latest_avsync_folder(base_path, tag="avsync15_va*"):
    #     """查找最新生成的avsync15_va开头的文件夹"""
    #     candidates = sorted(
    #         glob.glob(os.path.join(base_path, tag)),
    #         key=lambda x: os.path.getmtime(x),
    #         reverse=True
    #     )
    #     return candidates 

    # base_dir = "/root/workspace/bridiff_i2va/result"
    # image_dir = "/data_mount/vggsound/avsync15/videos_test_first_frame"
    # test_folders = find_latest_avsync_folder(base_dir)
    # test_folders_append = find_latest_avsync_folder("/root/workspace/bridiff_i2va/data", tag="repeat5*")
    # test_folders = test_folders + test_folders_append
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # for test_folder in test_folders:
    #     print(f"\n评测 {test_folder}")
    #     try:
    #         va_score, vi_score, ai_score = ib_va(
    #             video_dir=test_folder,
    #             audio_dir=test_folder,  # video和audio同目录
    #             image_dir=image_dir,
    #             batch_size=1,
    #             device=device
    #         )
            
    #         print("\n评测结果:")
    #         print(f"Video-Audio : {va_score:.4f}")
    #         print(f"Video-Image : {vi_score:.4f}")
    #         print(f"Audio-Image : {ai_score:.4f}")
            
    #         results = {
    #             "test_folder": os.path.basename(test_folder),
    #             "video_audio_score": f"{va_score:.4f}",
    #             "video_image_score": f"{vi_score:.4f}",
    #             "audio_image_score": f"{ai_score:.4f}",
    #             "batch_size": 1,
    #             "device_type": "GPU" if torch.cuda.is_available() else "CPU",
    #             "imagebind_version": "imagebind_huge"
    #         }            
    #         save_results(results, "./temp")  
            
    #     except Exception as e:
    #         print(f"评测失败: {str(e)}")
