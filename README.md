[Extrapolating from a Single Image to a Thousand Classes using Distillation](https://arxiv.org/abs/2112.00725)
---
by Yuki M. Asano* and Aaqib Saeed*  (*Equal Contribution)

![Our-method](https://single-image-distill.github.io/resources/animation_final.gif)

*Extrapolating from one image.* 
Strongly augmented patches from a single image are used to train a student (S) to distinguish semantic classes, such as those in ImageNet. 
The student neural network is initialized randomly and learns from a pretrained teacher (T) via KL-divergence. 
Although almost none of target categories are present in the image, we find student performances of **>66%** Top-1 Acc for classifying ImageNet's 1000 classes. 
In this paper, we develop this single datum learning framework and investigate it across datasets and domains.

## Key contributions

* A minimal framework for training neural networks with a single datum from scratch using distillation.
* Extensive ablations of the proposed method, such as the dependency on the source image, the choice of augmentations and network architectures.
* Large scale empirical evidence of neural networks' ability to extrapolate on > 13 image, video and audio datasets.
* Qualitative insights on what and how neural networks trained with a single image learn.

### Neuron visualizations
![Neurons](https://single-image-distill.github.io/resources/fig_7.png)

We compare activation-maximization-based visualizations using the [Lucent](https://github.com/greentfrapp/lucent) library.
Even though the model has never seen an image of a panda, the model trained with a teacher and only single-image inputs has a good idea of how a panda looks like.


# Running the experiments

### Installation
In each folder `cifar\in1k\video` you will find a requirements.txt file. Install packages as follows:
```
pip3 install -r requirements.txt
```

### 1. Prepare Dataset: 
To generate single image data, we refer to the [data_generation folder](data_generation)

### 2. Run Experiments:
There is a main "distill.py" file for each experiment type: small-scale and large-scale images and video.
Note: 2a uses tensorflow and 2b, 2c use pytorch.

#### 2a. Run distillation experiments for *CIFAR-10/100*
e.g. with Animal single-image dataset as follows:
```sh
# in cifar folder:
python3 distill.py --dataset=cifar10 --image=/path/to/single_image_dataset/ \
                   --student=wrn_16_4 --teacher=wrn_40_4 
```
Note that we provide a pretrained teacher model for reproducibility.

#### 2b. Run distillation experiments for *ImageNet* with single-image dataset as follows:
```sh
# in in1k folder:
python3 distill.py --dataset=in1k --testdir /ILSVRC12/val/ \
                   --traindir=/path/to/dataset/ --student_arch=resnet50 --teacher_arch=resnet18 
```
Note that teacher models are automatically downloaded from torchvision or timm. 


#### 2c. Run distillation experiments for *Kinetics* with single-image-created video dataset as follows:
```sh
# in video folder:
python3 distill.py --dataset=k400 --traindir=/dataset/with/vids --test_data_path /path/to/k400/val 
```
Note that teacher models are automatically downloaded from torchvideo when you distill a K400 model.

## Pretrained models
Large-scale (224x224-sized) image ResNet-50 models trained for 200ep:

| Dataset     | Teacher | Student | Performance | Checkpoint                                                                             |
|-------------|---------|---------|-------------|----------------------------------------------------------------------------------------|
| ImageNet-12 | R18     | R50     | 66.2%       | [R50 weights](https://www.dropbox.com/s/mo1d7n3im1aeyou/R50_from_R18_in1k.pth?dl=0)      |
| ImageNet-12 | R50     | R50     | 55.5%       | [R50 weights](https://www.dropbox.com/s/p1fskmmn96cksy7/R50_from_R50_in1k.pth?dl=0)      |
| Places365   | R18     | R50     | 50.3%       | [R50 weights](https://www.dropbox.com/s/i3dane5c60qw4d3/R50_from_R18_places365.pth?dl=0) |
| Flowers101  | R18     | R50     | 81.5%       | [R50 weights](https://www.dropbox.com/s/z5i17cw4u78iaz2/R50_from_R18_flowers101.pth?dl=0)   |
| Pets37      | R18     | R50     | 76.8%       | [R50 weights](https://www.dropbox.com/s/lxyhsne2jk6gi9h/R50_from_R18_pets37.pth?dl=0)      |
| IN100       | R18     | R50     | 66.2%       | [R50 weights](https://www.dropbox.com/s/jmtxm11o098dlc2/R50_from_R18_in100.pth?dl=0)     |
| STL-10      | R18     | R50     | 93.9%       | [R50 weights](https://www.dropbox.com/s/x0uk9g1zgnqlk3j/R50_from_R18_stl10.pth?dl=0)     |

Video x3d_s_e (expanded) models (160x160 crop, 4frames) trained for 400ep:

| Dataset | Teacher | Student  | Performance | Checkpoint                                                                              |
|---------|---------|----------|-------------|-----------------------------------------------------------------------------------------|
| K400    | x3d_xs  | x3d_xs_e | 51.8%       | [weights](https://www.dropbox.com/s/97tksbu9qty63z5/k400.pth?dl=0) |
| UCF101  | x3d_xs  | x3d_xs_e | 75.2%       | [weights](https://www.dropbox.com/s/4zeefz5jtzu9r01/ucf.pth?dl=0)  |


## Citation
```
@inproceedings{asano2021extrapolating,
  title={Extrapolating from a Single Image to a Thousand Classes using Distillation},
  author={Asano, Yuki M. and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2112.00725},
  year={2021}
}
```
