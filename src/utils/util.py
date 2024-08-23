%%writefile /content/AniPortrait/src/utils/util.py

import importlib
import os
import os.path as osp
import shutil
import sys
import cv2
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image,ImageDraw, ImageFont


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        codec = "libx264"
        container = av.open(path, "w")
        stream = container.add_stream(codec, rate=fps)

        stream.width = width
        stream.height = height

        for pil_image in pil_images:
            # pil_image = Image.fromarray(image_arr).convert("RGB")
            av_frame = av.VideoFrame.from_image(pil_image)
            container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
        container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")

###(modify)
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    #png logo
    png_path = "/content/drive/MyDrive/data/promethus_logo.png"
    png_image = Image.open(png_path).convert("RGBA")
    logo_size = (80,11)
    png_image = png_image.resize(logo_size) # logo_size

    font_path = "/content/drive/MyDrive/font/NanumSquareNeo-Variable.ttf"
    text= "이 프로젝트를 상업적으로 이용하거나 사적이익을 위해서 사용하지 않습니다."

    try:
        font = ImageFont.truetype(font_path, size=8)
    except IOError:
        font = ImageFont.load_default()

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        # Watermark - logo_png
        x.paste(png_image,  (width - logo_size[0] - 10, 10), png_image)
        # - text
        draw = ImageDraw.Draw(x)
        text_width, text_height = draw.textsize(text, font=font)
        text_position = ((width - text_width) // 2, height - text_height - 10)
        draw.text(text_position, text, (255, 255, 255), font=font)  # 흰색 텍스트 추가
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

def crop_face(img, lmk_extractor, expand=1.5):
    result = lmk_extractor(img)  # cv2 BGR

    if result is None:
        return None

    H, W, _ = img.shape
    lmks = result['lmks']
    lmks[:, 0] *= W
    lmks[:, 1] *= H

    x_min = np.min(lmks[:, 0])
    x_max = np.max(lmks[:, 0])
    y_min = np.min(lmks[:, 1])
    y_max = np.max(lmks[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    if width*height >= W*H*0.15:
        if W == H:
            return img
        size = min(H, W)
        offset = int((max(H, W) - size)/2)
        if size == H:
            return img[:, offset:-offset]
        else:
            return img[offset:-offset, :]
    else:
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        width *= expand
        height *= expand

        size = max(width, height)

        x_min = int(center_x - size / 2)
        x_max = int(center_x + size / 2)
        y_min = int(center_y - size / 2)
        y_max = int(center_y + size / 2)

        top = max(0, -y_min)
        bottom = max(0, y_max - img.shape[0])
        left = max(0, -x_min)
        right = max(0, x_max - img.shape[1])
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cropped_img = img[y_min + top:y_max + top, x_min + left:x_max + left]

    return cropped_img