import tensorflow as tf
import os
import json

import numpy as np

def load_image(pth, name):
    file = tf.io.read_file(os.path.join(pth, name))
    if name.lower().endswith('.png'):
        img = tf.io.decode_png(file, channels=3)
    else:
        img = tf.io.decode_jpeg(file, channels=3)
    return np.array(img)

# Define the features of the dataset
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature((), tf.int64),
    'width': tf.io.FixedLenFeature((), tf.int64),
    'channels': tf.io.FixedLenFeature((), tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

# Create a TFRecord file writer
tfrecord_filename = 'fruits.tfrecord'
writer = tf.io.TFRecordWriter(tfrecord_filename)

# Create a dictionary to map label strings to integer values
label_map = {}

# Loop through the dataset and serialize each record
dataset_dir = './fruits'
for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    for filename in os.listdir(label_dir):
        # Load the image and label
        image = load_image(label_dir, filename)
        if not(label in label_map):
            label_map[label] = len(label_map)
        label_int = label_map[label]
        [h,w,c] = image.shape

        # Serialize the data
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
            'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_int])),
        }))
        serialized_example = example.SerializeToString()

        # Write the serialized data to the TFRecord file
        writer.write(serialized_example)

# Close the writer
writer.close()

# Write the label map to a JSON file
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
