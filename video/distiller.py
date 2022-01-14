import os
import argparse
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import pytorchvideo
from pytorchvideo.models.x3d import create_x3d

from pytorch_lightning.plugins import DDPPlugin

import pytorchvideo.data
import torch.utils.data

import torchmetrics
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pl_bolts
import numpy as np
from collections import defaultdict

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    RandAugment, AugMix
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

# Trainer
class VideoDistill(pytorch_lightning.LightningModule):
    def __init__(self, dataset, teacher_ckpt, width_factor, depth_factor, warmup_epochs, epochs, lr, weight_decay):
        super().__init__()
        # models
        num_classes = 101 if dataset=='ucf' else 400
        self.teacher = create_teacher_model(num_classes=num_classes,
                                            teacher_ckpt=teacher_ckpt)
        self.student = create_student_model(num_classes=num_classes,
                                            width_factor=width_factor,
                                            depth_factor=depth_factor)

        # optimization
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay



        self.val_accuracy =  torchmetrics.Accuracy()



        self.batch_key = "video"
        self.temperature = args.temperature
        self.loss = nn.KLDivLoss(reduction='batchmean')

        self.teacher.eval()
        self.with_cutmix = True
        self.batch_size = args.batch_size

        for param in self.teacher.parameters():
            param.requires_grad = False

    def kd_loss_fn(self, outputs, teacher_outputs):
        T = self.temperature
        kd_loss = self.loss(F.log_softmax(outputs/T, dim=1),
                            F.softmax(teacher_outputs/T, dim=1))
        return kd_loss

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        if self.with_cutmix:
            with torch.no_grad():
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(x.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(x[:,0].size(), lam)
                x[:, :, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :,:, bbx1:bbx2, bby1:bby2]

        with torch.no_grad():
            teacher_predictions = self.teacher(x)
        student_predictions = self.student(x)
        loss = self.kd_loss_fn(student_predictions, teacher_predictions)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        y_hat = self.student(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val_acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        """
        This accumulated predictions per video-id, this way we can average them.
        Standard practice in video evaluations.
        """
        if batch_idx  == 0:
            self.accum_predictions = defaultdict(list)
            self.accum_labels = defaultdict()

        x = batch[self.batch_key]
        y_hat = self.student(x).softmax(dim=-1)
        for i in range(x.size(0)):
            self.accum_predictions[batch['video_index'][i].item()].append(y_hat[i].cpu())
            self.accum_labels[batch['video_index'][i].item()] = batch['label'][i].cpu()

        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test_acc", acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def test_epoch_end(self, test_step_outputs):
        """
        This finally yields the averaged accuracy
        """
        correct = []
        for k, y in self.accum_predictions.items():
            y_ = torch.stack(y).mean(0)
            correct.append(self.accum_labels[k] == y_.argmax(0))
        print(f"video_test_acc: {torch.tensor(correct).float().mean()*100:.2f}")
        self.log("video_test_acc", torch.tensor(correct).float().mean(),
                 on_step=False, on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.warmup_epochs, max_epochs=self.epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]



# Models
def create_student_model(width_factor, depth_factor, num_classes):
    return create_x3d(model_num_class=num_classes,
                      input_crop_size=160,
                      dropout_rate=0,
                      input_clip_length=4,
                      head_activation=None,
                      width_factor=width_factor,
                      depth_factor=depth_factor)

def create_teacher_model(num_classes, teacher_ckpt):
    if num_classes == 101:
        model = create_x3d(model_num_class=num_classes,
                           input_crop_size=160,
                           input_clip_length=4,
                           head_activation=None)
        ms = model.state_dict()
        pre_trained_state_dict = torch.load(teacher_ckpt, map_location='cpu')["state_dict"] # you need to pretrain this
        pre_trained_state_dict = {k.replace('model.', ''):v for k,v in pre_trained_state_dict.items() if v.shape == ms[k.replace('model.', '')].shape}
        model.load_state_dict(pre_trained_state_dict)
    else:
        model = create_x3d(model_num_class=num_classes,
                           input_crop_size=160, bottleneck_factor=2.25,
                           width_factor=2.0, depth_factor=2.2,
                           input_clip_length=4, head_activation=None)
        print([k for k,v in model.state_dict().items()])
        model = torch.hub.load("facebookresearch/pytorchvideo:main", model='x3d_xs', pretrained=True, head_activation=None)
    return model

# Datasets
class VideoDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, test_data_path, dist_data_path, batch_size, workers):
        super().__init__()
        self.mean = (0.45, 0.45, 0.45)
        self.std = (0.225, 0.225, 0.225)
        self.crop_size = 160

        self.side_size = 183
        self.num_frames = 4

        self.sampling_rate = 1
        self.frames_per_second = 12
        self.clip_duration = (self.num_frames * self.sampling_rate)/self.frames_per_second

        self.sampling_rate_val = 12
        self.clip_duration_val = (self.num_frames * self.sampling_rate_val)/self.frames_per_second

        self.test_data_path = test_data_path
        self.dist_data_path = dist_data_path

        self.batch_size = batch_size
        self.num_workers = workers

    # Dummy videos dataset
    def train_dataloader(self):
        train_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: (AugMix()(x.permute(1,0,2,3))).permute(1,0,2,3)),
                    Lambda(lambda x: x / 255.0),
                    Normalize(self.mean, self.std),
                    RandomShortSideScale(min_size=183, max_size=229),
                    RandomCrop(self.crop_size),
                    RandomHorizontalFlip(p=0.5),
                ])),
        ])
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dist_data_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            decode_audio=False,
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

    def val_dataloader(self):
        val_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
                ]))])
        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.test_data_path, 'val'),
            clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.clip_duration_val, 1),
            decode_audio=False,
            transform=val_transform
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        test_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
                ]))])
        test_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.test_data_path, 'val'),
            clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.clip_duration_val, 10),
            decode_audio=False,
            transform=test_transform
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

# Other helper functions below:

class CheckpointEveryNEpoch(pytorch_lightning.Callback):
    def __init__(self, every, save_path):
        self.every = every
        self.save_path = save_path

    def on_epoch_end(self, trainer: pytorch_lightning.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch % self.every == 0:
            ckpt_path = f"{self.save_path}/ckpt_{epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2