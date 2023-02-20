import tensorflow as tf
import os


# Define the features of the dataset
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

# Create a TFRecord file writer
tfrecord_filename = 'my_dataset.tfrecord'
writer = tf.io.TFRecordWriter(tfrecord_filename)

# Loop through the dataset and serialize each record
dataset_dir = 'path/to/my/dataset'
for filename in os.listdir(dataset_dir):
    # Load the image and label
    image = load_image(os.path.join(dataset_dir, filename))
    label = get_label(filename)

    # Serialize the data
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    serialized_example = example.SerializeToString()

    # Write the serialized data to the TFRecord file
    writer.write(serialized_example)

# Close the writer
writer.close()
