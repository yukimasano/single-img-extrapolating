# Single Image Pretraining of Visual Representations

_Note_: this is an edited version of the code from [here](https://github.com/yukimasano/single_img_pretraining), where we've added the video-generation code.
<p align="center">
<img src="https://single-image-distill.github.io/resources/patches.png" width="20%">
<img src="https://single-image-distill.github.io/resources/vid.gif" width="20%">
</p>

## Usage
For generating image-datasets run:
```sh
python make_single_img_dataset.py --imgpath images/ameyoko.jpg --targetpath ./out/ameyoko_dataset
# with 1.2M samples, the dataset will be roughly 100Gb large
```

For generating fake-video datasets from a single image run:
```sh
python make_fakevideo_dataset.py --imgpath images/ameyoko.jpg --targetpath ./out/ameyoko_video_dataset
# with 200K samples, the dataset will be roughly 10Gb large 
```

## Reference
If you find this code/idea useful, please consider citing the paper:
[**A critical analysis of self-supervision, or what we can learn from a single image**, Asano et al. ICLR 2020](https://arxiv.org/abs/1904.13132)
```
@inproceedings{asano2020a,
title={A critical analysis of self-supervision, or what we can learn from a single image},
author={Asano, Yuki M. and Rupprecht, Christian and Vedaldi, Andrea},
booktitle={International Conference on Learning Representations (ICLR)},
year={2020},
}
```
