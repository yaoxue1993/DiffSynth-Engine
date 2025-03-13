import imageio
import imageio.v3 as iio
import numpy as np
from PIL import Image
from typing import List


class VideoReader:
    def __init__(self, path: str):
        self.reader = imageio.get_reader(path)

    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()

    @property
    def frames(self) -> List[Image.Image]:
        return [self[i] for i in range(len(self))]


def load_video(path: str) -> VideoReader:
    return VideoReader(path)


def save_video(frames, save_path, fps=15):
    if save_path.endswith(".webm"):
        codec = "libvpx-vp9"
    elif save_path.endswith(".mp4"):
        codec = "libx264"

    frames = [np.array(img) for img in frames]

    # 使用 imageio 写入 .webm 文件
    with iio.imopen(save_path, "w", plugin="FFMPEG") as writer:
        writer.write(frames, fps=fps, codec=codec)
