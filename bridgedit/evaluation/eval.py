from tqdm import tqdm
import subprocess
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from paths import REPO_ROOT, BRIDGEDIT_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_video", type=str, default=None)
    parser.add_argument("--generated_video", type=str, default=os.path.join(BRIDGEDIT_ROOT, "save_videos/cross"))
    parser.add_argument("--target_audio", type=str, default=None)
    parser.add_argument("--generated_audio", type=str, default=os.path.join(BRIDGEDIT_ROOT, "save_videos/cross"))
    parser.add_argument("--seperate", type=bool, default=True)
    parser.add_argument("--video_caption_path", type=str,
                        default=os.path.join(REPO_ROOT, "caption_pipeline/recaption/avsync-test-72B-captions.json"))
    parser.add_argument("--audio_caption_path", type=str,
                        default=os.path.join(REPO_ROOT, "caption_pipeline/recaption/avsync-test-72B-captions.json"))
    parser.add_argument("--metrics", type=str, nargs='+', choices=['fvd', 'fad', 'clap', 'clipsim', 'ib', 'avalign','avalign_old', 'desync'], default=["fvd","fad","avalign_old"])
    args = parser.parse_args()
    print(args.metrics)
    if 'fvd' in args.metrics:
        from eval_fvd import Cal_FVD
        Cal_FVD(args.target_video, args.generated_video)
    if 'fad' in args.metrics:
        from eval_fad import Cal_FAD
        print("---- Calculating FAD Score ----")
        Cal_FAD(args.target_audio, args.generated_audio)
    if 'clap' in args.metrics:
        from eval_clap import Cal_Clap
        print("---- Calculating CLAP Score ----")
        # subprocess.run("conda activate clap")
        Cal_Clap(args.generated_audio, args.seperate, args.audio_caption_path, args.target_audio)
    if 'clipsim' in args.metrics:
        from eval_clipsim import Cal_Clipsim
        Cal_Clipsim(args.generated_video, args.seperate, args.video_caption_path)
    if 'ib' in args.metrics:
        from eval_ib_all import Cal_Ib
        Cal_Ib(args.generated_video, args.generated_audio, args.target_video, args.video_caption_path, args.audio_caption_path, args.seperate)
    if 'avalign' in args.metrics:
        from eval_avalign import Cal_Avalign
        Cal_Avalign(args.generated_video, args.generated_audio)
    if 'avalign_old' in args.metrics:
        from eval_avalign_old import Cal_Avalign_old
        Cal_Avalign_old(args.generated_video, args.generated_audio)
    if 'desync' in args.metrics:
        from eval_desync import Cal_Desync
        Cal_Desync(args.generated_video, args.generated_audio)

