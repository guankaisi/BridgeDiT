import logging
import os
from pathlib import Path

import torch
from colorlog import ColoredFormatter
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from av_bench.args import get_eval_parser
from av_bench.data.video_dataset import VideoDataset, error_avoidance_collate
from av_bench.synchformer.synchformer import Synchformer

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


def encode_video_with_sync(synchformer: Synchformer, x: torch.Tensor) -> torch.Tensor:
    # x: (B, T, C, H, W) H/W: 224

    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    # partition the video
    segment_size = 16
    step_size = 8
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
def extract(args):
    video_path: Path = args.video_path.expanduser()
    gt_audio: Path = args.gt_audio
    gt_cache: Path = args.gt_cache
    audio_length: float = args.audio_length
    num_workers: int = args.num_workers
    batch_size: int = args.gt_batch_size

    if gt_cache is None:
        if gt_audio is None:
            raise ValueError('Either gt_audio or gt_cache must be provided')
        gt_cache = gt_audio / 'cache'

    log.info('Extracting features...')

    # read all the file names
    video_names = os.listdir(video_path)
    video_paths = [video_path / f for f in video_names if f.endswith('.mp4')]
    log.info(f'{len(video_paths)} videos found.')

    dataset = VideoDataset(video_paths, duration_sec=4.0)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        collate_fn=error_avoidance_collate)

    sync_model = Synchformer().to(device).eval()
    sd = torch.load(_syncformer_ckpt_path, weights_only=True)
    sync_model.load_state_dict(sd)

    cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)
    
    output_sync_features = {}
    # output_ib_features = {}
    for data in tqdm(loader):
        name = data['name']
        # ib_video = data['ib_video'].to(device)
        sync_video = data['sync_video'].to(device)
        sync_features = cmp_encode_video_with_sync(sync_model, sync_video)
        

        sync_features = sync_features.cpu().detach()
        for i, n in enumerate(name):
            # saving a view will save the entire tensor so don't
            output_sync_features[n] = sync_features[i].clone()

    gt_cache.mkdir(parents=True, exist_ok=True)
    torch.save(output_sync_features, gt_cache / 'synchformer_video.pth')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = get_eval_parser()
    parser.add_argument('--video_path', type=Path, required=True, help='Path to the video files')
    args = parser.parse_args()
    extract(args)
