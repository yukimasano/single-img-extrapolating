Extrapolating from a Single Image to a Thousand Classes using Distillation
---
![Our-method](https://single-image-distill.github.io/resources/animation_final.gif)

*Extrapolating from one image.* 
Strongly augmented patches from a single image are used to train a student (S) to distinguish semantic classes, such as those in ImageNet. 
The student neural network is initialized randomly and learns from a pretrained teacher (T) via KL-divergence. 
Although almost none of target categories are present in the image, we find student performances of > 59% for classifying ImageNet's 1000 classes. 
In this paper, we develop this single datum learning framework and investigate it across datasets and domains.

## Key contributions

* A minimal framework for training neural networks with a single datum from scratch using distillation.
* Extensive ablations of the proposed method, such as the dependency on the source image, the choice of augmentations and network architectures.
* Large scale empirical evidence of neural networks' ability to extrapolate on > 13 image, video and audio datasets.
* Qualitative insights on what and how neural networks trained with a single image learn.
  
### Install Requirements:
In each folder `cifar\in1k\video` you will find a requirements.txt file. Install packages as follows:
```
pip3 install -r requirements.txt
```

### 1. Prepare Dataset: 
To generate single image data, we refer to the [data_generation folder](data_generation)

### 2. Run Experiments:
There is a main "distill.py" file for each experiment type: small-scale and large-scale images and video.

#### 2a. Run distillation experiments for *CIFAR-10/100*
e.g. with Animal single-image dataset as follows:
```sh
# in cifar folder:
python3 distill.py --dataset=cifar10 --image=/path/to/single_image_dataset/ --student=wrn_16_4 --teacher=wrn_40_4 
```
Note that we provide a pretrained teacher model for reproducibility.

#### 2b. Run distillation experiments for *ImageNet* with single-image dataset as follows:
```sh
# in in1k folder:
python3 distill.py --dataset=in1k --traindir=/path/to/dataset/ --testdir /ILSVRC12/val/ --save_dir=/path/to/savedir --student_arch=resnet50 --teacher_arch=resnet18 
```
Note that teacher models are automatically downloaded from torchvision or timm. 


#### 2c. Run distillation experiments for *Kinetics* with single-image-created video dataset as follows:
```sh
# in video folder:
python3 distill.py --dataset=k400 --traindir=/dataset/with/vids --save_dir=/path/to/savedir --teacher_arch=x3d_xs --test_data_path /path/to/k400/val
```
Note that teacher models are automatically downloaded from torchvideo when you distill a K400 model.


## Citation
```
@inproceedings{asano2021extrapolating,
  title={Extrapolating from a Single Image to a Thousand Classes using Distillation},
  author={Asano, Yuki M. and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2112.00725},
  year={2021}
}
```