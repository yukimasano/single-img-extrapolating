import numpy as np
import pytorch_lightning as pl
import timm.models as timm_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import AdamW

import utils


class ImgDistill(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 learning_rate,
                 weight_decay,
                 temperature,
                 maxepochs,
                 teacher_ckpt,
                 student_arch="resnet18",
                 teacher_arch="resnet50",
                 lr_schedule=True,
                 use_timm=False,
                 milestones=[100, 150]):
        super().__init__()

        if use_timm:
            self.teacher = timm_models.__dict__[teacher_arch](pretrained=num_classes == 1000, num_classes=num_classes)
        else:
            self.teacher = models.__dict__[teacher_arch](pretrained=num_classes == 1000, num_classes=num_classes)
        if teacher_ckpt != "":
            state_dict = torch.load(teacher_ckpt, map_location="cpu")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.teacher.load_state_dict(state_dict)
        if num_classes == 1000:
            self.student = models.__dict__[student_arch](pretrained=False, num_classes=num_classes)
        else:
            self.student = models.__dict__[student_arch](pretrained=False)
            self.student.fc = torch.nn.Linear(self.student.fc.weight.data.size(1), num_classes)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.lr_schedule = lr_schedule
        self.milestones = milestones

        self.maxepochs = maxepochs
        self.beta = 1.0
        self.with_cutmix = True
        self.loss = nn.KLDivLoss(reduction="batchmean")
        self.teacher.eval()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def kd_loss_fn(self, outputs, teacher_outputs):
        T = self.temperature
        kd_loss = self.loss(F.log_softmax(outputs / T, dim=1),
                            F.softmax(teacher_outputs / T, dim=1))
        return kd_loss

    def forward(self, x):
        y = self.student(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            if self.with_cutmix:
                lam = np.random.beta(self.beta, self.beta)
                rand_index = torch.randperm(x.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = utils.rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

        with torch.no_grad():
            teacher_predictions = self.teacher(x)
        student_predictions = self.student(x)
        loss = self.kd_loss_fn(student_predictions, teacher_predictions)
        self.log("train_loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        if self.lr_schedule:
            if self.trainer.is_last_batch:
                lr = self.learning_rate
                for milestone in self.milestones:
                    lr *= 0.5 if self.current_epoch >= milestone else 1.
                print(f"LR={lr}")
                print()
                for param_group in self.optimizers().param_groups:
                    param_group["lr"] = lr

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        student_predictions = self.student(x)
        loss = F.cross_entropy(student_predictions, y)
        topk = utils.accuracy(student_predictions, y, topk=(1, 5))
        same = (torch.argmax(student_predictions, dim=1) == y)
        acc = torch.sum(same) / float(y.size(0))

        self.log("val_acc", acc,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("val_loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("val_acc_top1", topk[0],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("val_acc_top5", topk[1],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.student.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        return [optimizer]
