import datasets
import models
import tensorflow as tf
import numpy as np

from tensorflow._api.v2 import io

from models import VGG


# Create a description of the features.
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)

raw_image_dataset = tf.data.TFRecordDataset('./datasets/fruits.tfrecord/')
parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))

model = VGG(weights='imagenet', include_top=True)

# 使用预训练模型进行预测
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
predictions = model.predict(img)