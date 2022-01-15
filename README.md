[Extrapolating from a Single Image to a Thousand Classes using Distillation](https://arxiv.org/abs/2112.00725)
---
by Yuki M. Asano* and Aaqib Saeed*  (*Equal Contribution)

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
| ImageNet-12 | R18     | R50     | 59.1%       | [R50 weights](https://www.dropbox.com/s/h13kgqdo5iqj8s7/in1k_r18_to_r50.pth?dl=0)      |
| ImageNet-12 | R50     | R50     | 53.5%       | [R50 weights](https://www.dropbox.com/s/gnfjpx8z4avkzlf/in1k_r50_to_r50.pth?dl=0)      |
| Places365   | R18     | R50     | 54.7%       | [R50 weights](https://www.dropbox.com/s/6idjbs3ig065a07/places365_r18_to_r50.pth?dl=0) |
| Flowers101  | R18     | R50     | 58.1%       | [R50 weights](https://www.dropbox.com/s/ho14zd8m3bwrhw9/flowers_r18_to_r50.pth?dl=0)   |
| Pets37      | R18     | R50     | 83.7%       | [R50 weights](https://www.dropbox.com/s/tatmrtv54t4v7an/pets_r18_to_r50.pth?dl=0)      |
| IN100       | R18     | R50     | 74.1%       | [R50 weights](https://www.dropbox.com/s/jm910bepij3eunp/in100_r18_to_r50.pth?dl=0)     |
| STL-10      | R18     | R50     | 93.0%       | [R50 weights](https://www.dropbox.com/s/t4k9yswcyp3a880/stl10_r18_to_r50.pth?dl=0)     |

Video x3d_s_e (expanded) models (160x160 crop, 4frames) trained for 400ep:

| Dataset | Teacher | Student  | Performance | Checkpoint                                                                              |
|---------|---------|----------|-------------|-----------------------------------------------------------------------------------------|
| K400    | x3d_xs  | x3d_xs_e | 53.57%      | [weights](https://www.dropbox.com/s/zbvtl14jakdltc6/k400-400ep-x3d_xs_wf5_df3.pth?dl=0) |
| UCF101  | x3d_xs  | x3d_xs_e | 77.32%      | [weights](https://www.dropbox.com/s/vy67dmk41z44c1t/ucf-400ep-x3d_xs_wf5_df3.pth?dl=0)  |


## Citation
```
@inproceedings{asano2021extrapolating,
  title={Extrapolating from a Single Image to a Thousand Classes using Distillation},
  author={Asano, Yuki M. and Saeed, Aaqib},
  journal={arXiv preprint arXiv:2112.00725},
  year={2021}
}
```
