import os
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


from distiller import VideoDistill, VideoDataModule, CheckpointEveryNEpoch

parser = argparse.ArgumentParser(description="Knowledge distillation of video models from a single image.")
# Distillation parameter
parser.add_argument("--temperature", default=5, type=float)

# Student model
parser.add_argument("--depth_factor", default=3.0, type=float)
parser.add_argument("--width_factor", default=5.0, type=float)

# Teacher model
parser.add_argument("--teacher_ckpt", default='/path/to/x3d_xs-teacher/ckpt', type=str)

# Optimizer settings
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float) # with 2 gpus
parser.add_argument("--weight_decay", default=0., type=float)

# Training settings
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--warmup_epochs", default=5, type=int)
parser.add_argument("--epochs", default=400, type=int)

# Dataset
parser.add_argument("--dataset", default='ucf or k400', choices=['ucf', 'k400'],
                    type=str, help="only used to choose teacher and num_classes")
parser.add_argument("--test_data_path", default='/path/to/UCF101/val', type=str)
parser.add_argument("--dist_data_path", default='/path/to/folder/train', type=str)
parser.add_argument("--workers", default=12, type=int)

# remainder
parser.add_argument("--eval_every", default=10, type=int)
parser.add_argument("--save_dir", default='./output/', type=str)


if __name__ == "__main__":
    args = parser.parse_args()
    # training module with teacher and student and optimizer
    distillation_module = VideoDistill(dataset=args.dataset,
                                       teacher_ckpt=args.teacher_ckpt,
                                       width_factor=args.width_factor,
                                       depth_factor=args.depth_factor,
                                       warmup_epochs=args.warmup_epochs,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       temperature=args.temperature,
                                       lr=args.lr, weight_decay=args.weight_decay)
    # data module with dataloaders
    data_module = VideoDataModule(test_data_path=args.test_data_path,
                                  dist_data_path=args.dist_data_path,
                                  batch_size=args.batch_size, workers=args.workers)

    # setup logging and saving dirs
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints')
    tensorboard_dir = f"./tensorboard/{args.save_dir.replace('/','-')[1:]}"
    tb_logger = TensorBoardLogger(save_dir=tensorboard_dir,
                                  name='video-distill', version='1')

    # setup trainer
    trainer = Trainer(gpus=-1, max_epochs=args.epochs,
                      callbacks=[LearningRateMonitor(), CheckpointEveryNEpoch(args.eval_every, checkpoint_path)],
                      logger=[tb_logger], check_val_every_n_epoch=args.eval_every,
                      progress_bar_refresh_rate=1, accelerator='ddp',
                      plugins=[DDPPlugin(find_unused_parameters=False)],
                      resume_from_checkpoint=checkpoint_path+'/last.ckpt' if os.path.isfile(checkpoint_path+'/last.ckpt') else False)
    # train
    trainer.fit(distillation_module, data_module)

    # do the 10-temporal-crop eval
    trainer.test(distillation_module, data_module)
