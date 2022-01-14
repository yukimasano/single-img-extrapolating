# Source: https://keras.io/examples/vision/cutmix/
import tensorflow as tf
IMG_SIZE = 32 

@tf.function
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)
    cut_w = IMG_SIZE * cut_rat  
    cut_w = tf.cast(cut_w, tf.int32)
    cut_h = IMG_SIZE * cut_rat  
    cut_h = tf.cast(cut_h, tf.int32)
    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  
    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)
    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1
    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1
    return boundaryx1, boundaryy1, target_h, target_w

@tf.function
def cutmix(x):
    image1 = x
    image2 = tf.reverse(x, axis=[0])
    alpha = [0.25]
    beta = [0.25]
    lambda_value = sample_beta_distribution(1, alpha, beta)
    lambda_value = lambda_value[0][0]
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)
    crop2 = tf.image.crop_to_bounding_box(image2, boundaryy1, boundaryx1, target_h, target_w)
    image2 = tf.image.pad_to_bounding_box(crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
    crop1 = tf.image.crop_to_bounding_box(image1, boundaryy1, boundaryx1, target_h, target_w)
    img1 = tf.image.pad_to_bounding_box(crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE)
    image1 = image1 - img1
    image = image1 + image2
    lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
    lambda_value = tf.cast(lambda_value, tf.float32)
    return image

    