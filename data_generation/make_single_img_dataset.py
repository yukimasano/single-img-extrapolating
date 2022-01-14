import argparse
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as tfs

import torch
import os
import random
import math
import multiprocessing
from joblib import Parallel,delayed
import time

from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

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
    def __init__(self, image_path, initcrop=0.5, degrees=30, scale=(0.01, 1.0), shear=30, vflip=False,
                  crop_size=256, mean=[0, 0, 0], std=[1, 1, 1], randinterp=False, cropfirst=True, debug=False,
                 length=int(1.281e6)):
        """
        Dataset class that combines the following transformations to generate samples from A SINGLE IMAGE
        1. RandomCrop (initialcrop) if cropfirst
        2. Random-Resized-Crop  if scale[0] > 0
            * size=(sqrt(2)*cropsize)            # sqrt(2) in the case of 45deg rotation.
            * scale=(1/scale[0], 1/scale[1])
        3. RandomAffine (degrees and shear) if shear!=0 and degrees!=0
        4. VerticalFlip (p=0.5) if vflip
        5. RandomHorizontalflip (p=0.5)
        6. CenterCrop (cropsize)
        7. ColorJitter
        #  8. RandomGaussianBlur # from MoCo-v2 repo
        9. ...
        
        
        """
        self.img = None
        self.image_path = image_path
        normalize = tfs.Normalize(mean=mean, std=std)
        self.img = (Image.open(self.image_path).convert('RGB'))

        scale = (1./scale[0], 1./scale[1])
        if randinterp and not debug:
            resizedcrop = tfs.RandomApply([MyRandomResizedCrop(int(math.sqrt(2) * crop_size), scale=scale, interpolation=1),
                                           MyRandomResizedCrop(int(math.sqrt(2) * crop_size), scale=scale, interpolation=2),
                                           MyRandomResizedCrop(int(math.sqrt(2) * crop_size), scale=scale, interpolation=3)])
        else:
            resizedcrop = MyRandomResizedCrop(int(math.sqrt(2) * crop_size), scale=scale, interpolation=3)
        self.length = length # run out after one IN-1k epoch
        self.crop_size = crop_size


        tfslist = []
        if cropfirst:
            tfslist.append(tfs.RandomCrop(int(initcrop*min(self.img.size))))
          
        if scale[0] > 0 :
            tfslist.append(resizedcrop)
        if shear != 0:
            tfslist.append(tfs.RandomAffine(degrees, translate=None, scale=None, shear=shear, resample=Image.BILINEAR, fillcolor=0))
        elif degrees != 0:
            tfslist.append(tfs.RandomRotation(degrees=degrees, resample=Image.BILINEAR))

        if vflip:
            tfslist.append(tfs.RandomVerticalFlip(p=0.5))

        tfslist.append(tfs.CenterCrop(crop_size))
        if debug:
            tfslist.append(tfs.Resize((32, 32)))
        else:
            tfslist.extend([
                tfs.RandomApply([tfs.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], 0.5),
                # GaussianBlur()
            ])
        tfslist.append(tfs.ToTensor())
        if mean != [0, 0, 0]:
            tfslist.append(normalize)
        self.transforms = tfs.Compose(tfslist)
        print("transforms:", tfslist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.img is None:
            self.img = Image.open(self.image_path).convert('RGB')
        return self.transforms(self.img)


def get_mono_data_loader(batch_size, num_workers, **kwargs):
    m = MonoDataset(**kwargs)
    return torch.utils.data.DataLoader(m, batch_size=batch_size, num_workers=num_workers,
                                       drop_last=True)

def save_batch_imgs(imgs, count, pth):
    for k in range(imgs.size(0)):
        save_image(imgs[k,:,:,:], pth+'/patch_%s.jpeg'%(k+count), padding=0,normalize=False, scale_each=False)


def make_(thred):
    random.seed(555*thred)
    torch.manual_seed(555*thred)
    data_loader = get_mono_data_loader(args.batch_size, 0,
                                       image_path=args.imgpath,
                                       initcrop=args.initcrop, degrees=args.deg,
                                       scale=tuple(args.scale), shear=args.shear,
                                       vflip=args.vflip,
                                       crop_size=args.img_size, randinterp=args.randinterp,
                                       cropfirst=args.cropfirst, length=args.img_per_thread,
                                       )
    count = thred*args.img_per_thread
    print(f"{len(data_loader.dataset) + (args.img_per_thread % args.batch_size)}images per thread")
    for patches in data_loader:
        count += len(patches)
        save_batch_imgs(patches, count, path)

    # last batch: treat specially
    diter = iter(data_loader)
    patches = next(diter)[:(args.img_per_thread % args.batch_size)]
    count += len(patches)
    save_batch_imgs(patches, count, path)
    return 0


def get_parser():
    parser = argparse.ArgumentParser(description='Single Image Pretraining, Asano et al. 2020')
    # Generation settings
    parser.add_argument('--img_size', default=32, type=int, help='Size of generated images (default:256)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batchsize for generation (default:32)')
    parser.add_argument('--num_imgs', default=50000, type=int, help='Number of images to be generated (default: 1281167)')
    parser.add_argument('--threads', default=20, type=int, help='how many CPU threads to use for generation (default: 20)')

    # Flipping
    parser.add_argument('--vflip', action='store_true', help='use vflip? (default: False)')

    # Geometric augmentations
    parser.add_argument('--deg', default=30, type=float, help='max rot angle (default: 30)')
    parser.add_argument('--shear', default=30, type=float, help='max shear angle (default: 30)')

    # note: we don't do color augmentation or hflip because that's usually in the pretraining method's codebase already.

    # Scale/Crop augmentations
    parser.add_argument('--cropfirst', action='store_false', help='usage of initial crop to not focus too much on center (default: True)')
    parser.add_argument('--initcrop', default=0.5, type=float, help='initial crop size relative to image (default: 0.5)')
    parser.add_argument('--scale', default=[500, 1], nargs=2, type=float, help='data augmentation inverse scale (default: 500, 1)')
    parser.add_argument('--randinterp', action='store_true',
                        help='For RR crops: use random interpolation method or just bicubic? (default: False)')

    # storing etc.
    parser.add_argument('--debug', default=False)
    parser.add_argument('--imgpath', default="images/ameyoko.jpg", type=str)
    parser.add_argument('--targetpath', default="./out", type=str)
    return parser


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    args = get_parser().parse_args()
    if args.debug:
        args.num_imgs = 64
        args.threads = 2

    path = args.targetpath + str(args.img_size) + "_single" + args.imgpath.split('/')[-1].split('.')[0]
    path += f"_init{args.initcrop}_deg{args.deg}_scale{args.scale[0]}_{args.scale[1]}"
    path += f"_shear{args.shear}_randinterp_{args.randinterp}_vflip{args.vflip}_cropfirst{args.cropfirst}_new_{args.num_imgs}"
    path += '/train/dummy'
    os.makedirs(path, exist_ok=True)
    args.img_per_thread = args.num_imgs//args.threads

    print(args)
    print(f"will save {args.num_imgs } patches in {path}", flush=True)

    time.sleep(1)
    t0 = time.time()
    q = Parallel(n_jobs=args.threads)(
        delayed(make_)(i)
        for i in range(args.threads)
    )

    print(f"{args.num_imgs} took {(time.time()-t0)/60:.2f}min with {args.threads} threads", flush=True)
    if args.debug:
        tensors = []
        for i in os.listdir(path):
            p = Image.open(path+"/"+i)
            tensors.append(tfs.ToTensor()(p))
        save_image(tensors, path.replace('/dummy', '.png'), nrow=8, padding=3)

    ############################ 10 imagenet images ########################################
    # chosen = ['n04443257/n04443257_36457.JPEG',
    #           'n02129604/n02129604_19761.JPEG', 'n03733281/n03733281_41945.JPEG',
    #           'n02727426/n02727426_50177.JPEG', 'n07753592/n07753592_3046.JPEG',
    #           'n03485794/n03485794_27904.JPEG', 'n03220513/n03220513_16200.JPEG',
    #           'n04487394/n04487394_23489.JPEG',
    #           'n07614500/n07614500_271.JPEG', 'n02105412/n02105412_3932.JPEG',
    #           ]