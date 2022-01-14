import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import tensorflow as tf
import data
import network
import distiller

parser = argparse.ArgumentParser(
  description="Knowledge Distillation From a Single Image.")
parser.add_argument("--epochs", 
  default=1000, type=int, 
  help="number of total epochs to run")
parser.add_argument("--batch_size", 
  default=512, type=int,
  help="batch size")
parser.add_argument("--learning_rate", 
  default=0.001, type=float, 
  help="initial learning rate")
parser.add_argument("--temperature",
  default=8, type=float, 
  help="temperature for distillation")
parser.add_argument("--images_dir", 
  default="./datasets", type=str, 
  help="directory holding distillation images")
parser.add_argument("--model_dir", 
  default="./distilled_models/", type=str, 
  help="model directory path")
parser.add_argument("--tfb_log_dir", 
  default="./logs/", type=str, 
  help="logs directory path")
parser.add_argument("--image", type=str, 
  default="animals",
  help="image directory to use for distillation")
parser.add_argument("--dataset", type=str, 
  default="cifar10",
  help="dataset name to use for evaluation")
parser.add_argument("--teacher", type=str, 
  help="model name to use as teacher")
parser.add_argument("--student", type=str, 
  help="model name to use as student")
args = parser.parse_args()

DATASETS = {"cifar10": data.get_cifar_10, 
  "cifar100": data.get_cifar_100}

MODELS = {
  "wrn_16_4": network.wrn_16_4_network,
  "wrn_40_4": network.wrn_40_4_network,
}

def main(args):
  data_fn = DATASETS[args.dataset]
  _, test_ds, num_classes = data_fn()
  
  images_dir = os.path.join(args.images_dir,args.image)
  tr_dist_data = data.create_distill_data(images_dir, args.batch_size)

  teacher_fn = MODELS[args.teacher]
  teacher = teacher_fn(model_dir=args.dataset,
    num_classes=num_classes, 
    load_weights=True)
  teacher.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  teacher.evaluate(test_ds, verbose=2)

  student_fn = MODELS[args.student]
  student = student_fn(model_dir=args.dataset,
    num_classes=num_classes, 
    load_weights=False)

  trainer = distiller.Distiller(student, teacher)
  trainer.compile(optimizer=tf.keras.optimizers.Adam(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    temperature=args.temperature)

  checkpoint_filepath = os.path.join(args.model_dir, args.dataset, 
    f"teacher_{args.teacher}_student_{args.student}_{args.image}_temp_{args.temperature}")
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="distillation_loss",
    save_weights_only=True)
  tb_log_dir = os.path.join(args.tfb_log_dir, 
    f"{args.dataset}_teacher_{args.teacher}_student_{args.student}_{args.image}_temp_{args.temperature}/") 
  if not os.path.exists(tb_log_dir):
    os.makedirs(tb_log_dir)
  tb_callback = tf.keras.callbacks.TensorBoard(tb_log_dir, update_freq=1)

  trainer.fit(tr_dist_data, epochs=args.epochs, verbose=2,
    callbacks=[model_checkpoint_callback, tb_callback])

  student_model = trainer.student
  student_model.compile(metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  student.evaluate(test_ds, verbose=2)

if __name__ == "__main__":
  main(args)