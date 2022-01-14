#pip install moviepy

import argparse
from PIL import Image
from torchvision.utils import save_image, make_grid
import torchvision.transforms as tfs
import torchvision.transforms.functional as F
from moviepy.editor import *

import torch
import os
import random
import math
import multiprocessing
from joblib import Parallel,delayed
import time
import random

from PIL import ImageFilter
import random
import numpy as np
import matplotlib.pyplot as plt

from pytorchvideo.transforms import AugMix

from moviepy.editor import ImageSequenceClip

class MyRandomResizedCrop(tfs.RandomResizedCrop):
    @staticmethod
    def get_params(
            img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        # compared to original: there's no fallback of doing a CentralCrop

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size[0], img.size[1]
        area = height * width

        while True: #  basically no falling back to center-crop, as in the original implementation
            target_area = area * (
                torch.empty(1).uniform_(scale[0], scale[1]).item()    # **2 # the cubed is new
            )
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

class MonoDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, initcrop=0.5, scale=(0.01, 1.0), vflip=True, hflip=True,
                 crop_size=256, mean=[0, 0, 0], std=[1, 1, 1], cropfirst=True, frames=8,
                 length=int(1.281e6)):
        self.frames = frames

        self.img = None
        self.image_path = image_path
        normalize = tfs.Normalize(mean=mean, std=std)
        self.img = (Image.open(self.image_path).convert('RGB'))

        self.scale = (1./scale[0], 1./scale[1])
        self.resizedcrop = MyRandomResizedCrop(int(math.sqrt(2) * crop_size), scale=self.scale, interpolation='bicubic')
        self.length = length # run out after one  epoch
        self.crop_size = crop_size
        self.augmix = AugMix()



        if cropfirst:
            self.init_crop = tfs.RandomCrop(int(initcrop*min(self.img.size)))

        if vflip:
            self.vflip = tfs.RandomVerticalFlip(p=0.5)
        if hflip:
            self.hflip = tfs.RandomHorizontalFlip(p=0.5)

        tfslist = []
        tfslist.append(tfs.CenterCrop(self.crop_size))
        self.cj = tfs.RandomApply([tfs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], 0.5)
        tfslist.append(tfs.ToTensor())
        if mean != [0, 0, 0]:
            tfslist.append(normalize)
        self.transforms = tfs.Compose(tfslist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        params = []
        if self.img is None:
            self.img = Image.open(self.image_path).convert('RGB')
        img = self.init_crop(self.img)
        img = self.cj(img)

        steps = 2
        for _ in range(steps):
            params.append(self.resizedcrop.get_params(img, scale=self.scale, ratio=(3. / 4., 4. / 3.)))
        i,j,h,w = params[0]
        i2,j2,h2,w2 = params[1]
        i_frames, j_frames = np.linspace(i,i2,self.frames),np.linspace(j,j2,self.frames)
        h_frames, w_frames = np.linspace(h,h2,self.frames),np.linspace(w,w2,self.frames)
        frames = []
        for f in range(self.frames):
            frames.append(F.resized_crop(img, i_frames[f], j_frames[f], h_frames[f], w_frames[f],
                                         (self.crop_size, self.crop_size), Image.BICUBIC))
        return torch.cat([self.transforms(f).unsqueeze(0) for f in frames],dim=0)

def get_mono_data_loader(batch_size, num_workers, **kwargs):
    m = MonoDataset(**kwargs)
    return torch.utils.data.DataLoader(m, batch_size=batch_size, num_workers=num_workers,
                                       drop_last=True)

def save_batch_vids(imgs, count, pth, thread):
    all_files = []
    for k in range(imgs.size(0)):
        files = []
        for frame in range(imgs.size(1)):
            name = pth+f'/_tmp_patch_{k+count}_f{frame}.png'
            save_image(imgs[k,frame,:,:], name, padding=0,
                       normalize=False, scale_each=False)
            files.append(name)
            all_files.append(name)
        clip = ImageSequenceClip(files, fps = 12)
        clip.write_videofile(pth+ f"clips/clip_{thread}_{k+count}.mp4", fps = 15)
    for f in all_files:
        os.remove(f)

def make_(thred, args):
    random.seed(555*thred)
    torch.manual_seed(555*thred)
    data_loader = get_mono_data_loader(args.batch_size, 0,
                                       image_path=args.imgpath,
                                       initcrop=args.initcrop,
                                       scale=tuple(args.scale),
                                       vflip=not args.no_vflip,
                                       crop_size=args.img_size,
                                       cropfirst=args.cropfirst, length=args.img_per_thread,
                                       )
    count = thred*args.img_per_thread
    print(f"{len(data_loader.dataset) + (args.img_per_thread % args.batch_size)}images per thread")
    for patches in data_loader:
        count += len(patches)*args.frames
        save_batch_vids(patches, count, path,thred)

    # last batch: treat specially
    diter = iter(data_loader)
    patches = next(diter)[:(args.img_per_thread % args.batch_size)]
    count += len(patches)
    save_batch_vids(patches, count, path, thred)
    return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Single Image Pretraining, Asano et al. 2020')
    # Generation settings
    parser.add_argument('--img_size', default=230, type=int, help='Size of generated images (default:256)')
    parser.add_argument('--batch_size', default=50, type=int, help='Batchsize for generation (default:32)')
    parser.add_argument('--num_imgs', default=200000, type=int, help='Number of images to be generated (default: 1281167)')
    parser.add_argument('--threads', default=20, type=int, help='how many CPU threads to use for generation (default: 20)')

    # Flipping
    parser.add_argument('--no_vflip', action='store_false', help='use vflip? (default: True)')


    # Scale/Crop augmentations
    parser.add_argument('--cropfirst', action='store_false', help='usage of initial crop to not focus too much on center (default: True)')
    parser.add_argument('--initcrop', default=0.5, type=float, help='initial crop size relative to image (default: 0.5)')
    parser.add_argument('--scale', default=[500, 1], nargs=2, type=float, help='data augmentation inverse scale (default: 500, 1)')

    # video stuff
    parser.add_argument('--frames', default=16, type=int, help='number of frames') # fps is fixed to 12

    # storing etc.
    parser.add_argument('--imgpath', default="images/img_b.jpg", type=str)
    parser.add_argument('--targetpath', default="/scratch/local/nvme/yuki/data/kd/", type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    args.img_per_thread = args.num_imgs//args.threads
    random.seed(555)
    torch.manual_seed(555)
    path = args.targetpath +"/" + str(args.img_size) + "_single" + args.imgpath.split('/')[-1].split('.')[0]
    path += f"_init{args.initcrop}_scale{args.scale[0]}_{args.scale[1]}"
    path += f"_novflip{args.no_vflip}_cropfirst{args.cropfirst}_new_{args.num_imgs}_noaugmix_frames_{args.frames}"
    path += '/train/'
    os.makedirs(path+"clips", exist_ok=True)
    args.img_per_thread = (args.num_imgs//args.threads)

    time.sleep(1)
    t0 = time.time()
    q = Parallel(n_jobs=args.threads)(
        delayed(make_)(i,args)
        for i in range(args.threads)
    )

