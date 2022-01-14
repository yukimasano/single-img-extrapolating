import tensorflow as tf

class Distiller(tf.keras.Model):
  def __init__(self, student, teacher, 
    has_labels=False):
    super(Distiller, self).__init__()
    self.teacher = teacher
    self.student = student
    self.source_dataset_has_labels = has_labels

  def compile(self,
    optimizer, metrics,
    distillation_loss_fn, 
    temperature):
    super(Distiller, self).compile(
      optimizer=optimizer, metrics=metrics)
    self.distillation_loss_fn = distillation_loss_fn
    self.temperature = temperature

  def train_step(self, data):
    if self.source_dataset_has_labels:
      # If knowledge distillation from source data with labels
      x, _ = data
    else:
      x = data # If knowledge distillation from single image source
    teacher_predictions = self.teacher(x, training=False)
    with tf.GradientTape() as tape:
        student_predictions = self.student(x, training=True)
        loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
            tf.nn.softmax(student_predictions / self.temperature, axis=1))
    trainable_vars = self.student.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    results = {m.name: m.result() for m in self.metrics}
    results.update({"distillation_loss": loss})
    return results

  def test_step(self, data):
    x, y = data
    y_prediction = self.student(x, training=False)
    self.compiled_metrics.update_state(y, y_prediction)
    results = {m.name: m.result() for m in self.metrics}
    return results