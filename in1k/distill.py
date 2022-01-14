import os
import argparse

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytorch_lightning import  Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

import utils
from distiller import ImgDistill

CLASSES = {"in1k": 1000, "pets37": 37, "flowers102":102, "stl10":10, "places365": 365}


parser = argparse.ArgumentParser(description="Knowledge Distillation From a Single Image.")

# Teacher settings
parser.add_argument("--teacher_arch",default="resnet18", type=str, help="arch for teacher")
parser.add_argument("--use_timm", action="store_true", help="use strong-aug trained timm models?")
parser.add_argument("--teacher_ckpt", default="", type=str, help="ckpt to load teacher. not needed for IN-1k")

# Student
parser.add_argument("--student_arch", default="resnet50", type=str, help="arch for student")
parser.add_argument("--temperature", default=8, type=float, help="temperature logits are divided by")

# Training settings
parser.add_argument("--lr_schedule", action="store_true", help="lr_schedule")
parser.add_argument("--milestones", default=[100, 150], nargs="*", type=int, help="lr schedule (drop lr by 5x)")
parser.add_argument("--epochs", default=200, type=int, help="number of total epochs to run")
parser.add_argument("--batch_size", default=512, type=int, help="batch size per GPU")

# Optimizer
parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
parser.add_argument("--wd", default=5e-4, type=float, help="weight decay")

# data
parser.add_argument("--traindir", default="/tmp/train/", type=str, help="folder with folder(s) of training imgs")
parser.add_argument("--testdir", default="/datasets/ILSVRC12/val/", type=str, help="folder with folder(s) of test imgs")

# saving etc.
parser.add_argument("--save_dir",default="save_dir/", type=str, help="saving dir")
parser.add_argument("--dataset", default="in1k", type=str, help="dataset name -- for saving and choosing num_classes")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--save_every", default=10, type=int, help="save every n epochs")
parser.add_argument("--eval_every", default=1, type=int, help="save every n epochs")
parser.add_argument("--tensorboard_dir", default="./tensorboard/kd", type=str, help="directory for tensorboard ")


if __name__ == "__main__":
    args = parser.parse_args()

    # Define training augmentations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentations = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.), interpolation=3),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
    # Define training dataset
    train_dataset = datasets.ImageFolder(
        args.traindir,
        transforms.Compose(augmentations))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    # Define eval augmentations
    transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    # Define eval dataset
    val_dataset = datasets.ImageFolder(
        args.testdir,
        transforms.Compose(transform))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    # setup logging and saving dirs
    checkpoint_path = os.path.join(args.save_dir, version)
    tb_logger = TensorBoardLogger(save_dir=args.tensorboard_dir,
                    name=args.dataset, version='1')
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, 
                            monitor="val_acc",
                            save_last=True, 
                            filename=f"best_{args.dataset}")

    # training module with teacher and student and optimizer
    distiller = ImgDistill(
                            num_classes=CLASSES[args.dataset],
                            learning_rate=args.lr,
                            weight_decay=args.wd,
                            temperature=args.temperature,
                            maxepochs=args.epochs,
                            teacher_ckpt=args.teacher_ckpt,
                            student_arch=args.student_arch,
                            lr_schedule=args.lr_schedule,
                            teacher_arch=args.teacher_arch,
                            use_shampoo=args.use_shampoo,
                            use_timm=args.use_timm)
    # setup trainer
    trainer = Trainer(gpus=-1, max_epochs=maxepochs,
                        callbacks=[checkpoint_callback,
                                   utils.CheckpointEveryEpoch(args.save_every, checkpoint_path)],
                        logger=[tb_logger], check_val_every_n_epoch=args.eval_every,
                        progress_bar_refresh_rate=1, accelerator="ddp",
                        plugins=[DDPPlugin(find_unused_parameters=False)])
    # train
    trainer.fit(distiller, train_loader, val_loader)
