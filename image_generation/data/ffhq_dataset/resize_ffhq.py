import os
import torch
import sys
import torchvision.transforms.functional as TF
import argparse

from cleanfid.resize import make_resizer
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Resize FFHQ images.')
parser.add_argument('--img_size', type=int, help='Size of the images after resizing')
args = parser.parse_args()


def resize_ffhq():
    img_size = args.img_size
    fn_resize = make_resizer("PIL", False, "bicubic", (img_size, img_size))
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    root_dir = 'thumbnails128x128'
    savedir = 'ffhq{}'.format(img_size)
    
    # count = 0
    for subdir, _, files in tqdm(os.walk(root_dir)):
        for file in files:
            if file.endswith(".png"):
                dir_name = os.path.join(savedir, '{}000'.format(file[:2]))
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                save_path = os.path.join(savedir, '{}000'.format(file[:2]), file)
                if not os.path.exists(save_path):
                    path = os.path.join(subdir, file)
                    img = transform(Image.open(path))
                    img = torch.clip(img*255., 0, 255).to(torch.uint8)
                    img_np = img.numpy().transpose((1, 2, 0))
                    img_res = fn_resize(img_np)
                    img = torch.tensor(img_res.transpose((2, 0, 1))) / 255.
                    img_pil = TF.to_pil_image(img)
                    # 保存
                    img_pil.save(save_path)


if __name__ == '__main__':    
    resize_ffhq()
