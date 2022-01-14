import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from cutmix_utils import cutmix
AUTO = tf.data.experimental.AUTOTUNE

STD = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
MEAN = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))

def cutout(images, labels):
  _images = tfa.image.cutout(images, mask_size=(16,16))
  return _images, labels

def mixup(x):
  # From: https://keras.io/examples/keras_recipes/better_knowledge_distillation/
  alpha = tf.random.uniform([], 0, 1)
  mixedup_x = alpha * x + (1 - alpha) * tf.reverse(x, axis=[0])
  return mixedup_x

def train_prep_cifar(x, y):
  x = tf.cast(x, tf.float32) / 255.
  x = tf.image.random_flip_left_right(x)
  x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
  x = tf.image.random_crop(x, (32, 32, 3))
  x = (x - MEAN) / STD
  return x, y

def valid_prep_cifar(x, y):
  x = tf.cast(x, tf.float32) / 255.
  x = (x - MEAN) / STD
  return x, y

def get_cifar_10(batch_size = 128, shuffle_buffer = 10000, with_cutout=True):
  ds = tfds.load("cifar10", as_supervised=True)
  train_ds = ds["train"].map(train_prep_cifar, AUTO).shuffle(shuffle_buffer).batch(batch_size)
  if with_cutout:
    train_ds = train_ds.map(cutout, AUTO)
  train_ds = train_ds.prefetch(-1)
  test_ds = ds["test"].map(valid_prep_cifar).batch(batch_size * 4).prefetch(-1)
  return train_ds, test_ds, 10

def get_cifar_100(batch_size = 128, shuffle_buffer = 10000, with_cutout=True):
  ds = tfds.load("cifar100", as_supervised=True)
  train_ds = ds["train"].map(train_prep_cifar, AUTO).shuffle(shuffle_buffer).batch(batch_size)
  if with_cutout:
    train_ds = train_ds.map(cutout, AUTO)
  train_ds = train_ds.prefetch(-1)
  test_ds = ds["test"].map(valid_prep_cifar).batch(batch_size * 4).prefetch(-1)
  return train_ds, test_ds, 100

def augment_distill(x):
  x = tf.cast(x, tf.float32)/255.
  x = tf.image.random_flip_left_right(x)
  x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
  x = tf.image.random_crop(x, (32, 32, 3))
  x = (x - MEAN) / STD
  return x

def augment_distill_map(x):
  return tf.map_fn(augment_distill, x)

def create_distill_data(data_directory, batch_size):
  distill_data = tf.keras.preprocessing.image_dataset_from_directory(data_directory,
    label_mode=None, shuffle=True, batch_size=batch_size, image_size=(32,32))
  distill_data = distill_data.map(augment_distill_map, 
    num_parallel_calls=AUTO).shuffle(10)
  distill_data = distill_data.map(cutmix, 
    num_parallel_calls=AUTO)
  distill_data = distill_data.prefetch(AUTO)
  return distill_data