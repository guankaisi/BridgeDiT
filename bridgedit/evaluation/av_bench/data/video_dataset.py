import logging
from pathlib import Path

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder

from av_bench.data.ib_data import SpatialCrop

log = logging.getLogger()

# https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
# https://pytorchvideo.readthedocs.io/en/latest/_modules/pytorchvideo/transforms/functional.html
_IMAGEBIND_SIZE = 224
_IMAGEBIND_FPS = 0.5

_SYNC_SIZE = 224
_SYNC_FPS = 10.0


def error_avoidance_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class VideoDataset(Dataset):

    def __init__(
        self,
        video_paths: list[Path],
        *,
        duration_sec: float = 5.0,
    ):
        self.video_paths = video_paths

        self.duration_sec = duration_sec

        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)
        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.crop = SpatialCrop(_IMAGEBIND_SIZE, 3)

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]

        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_IMAGEBIND_FPS * self.duration_sec),
            frame_rate=_IMAGEBIND_FPS,
            format='rgb24',
        )
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()
        sync_chunk = data_chunk[1]
        
        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_path}')
        if sync_chunk.shape[0] < self.sync_expected_length:
            # raise RuntimeError(
            #     f'Sync video too short {video_path}, expected {self.sync_expected_length}, got {sync_chunk.shape[0]}'
            # )
            repeat_times = int(self.sync_expected_length // sync_chunk.shape[0]) + 1
            repeat_times_tuple = (repeat_times,) + (1,) * (sync_chunk.ndimension() - 1)  # 只沿着dim=0重复
            sync_chunk = sync_chunk.repeat(repeat_times_tuple)

        

        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_path}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'name': video_path.stem,
            'sync_video': sync_chunk,
        }

        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.video_paths[idx]}: {e}')
            return None

    def __len__(self):
        return len(self.video_paths)
