import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class AlexNet(Model):
    def __init__(self, num_classes, *args, **kwargs):
        super(self,AlexNet).__init__(*args, **kwargs)
        self.model = Sequential([
            # 第一层卷积
            Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
            MaxPooling2D((3, 3), strides=(2, 2)),
            # 第二层卷积
            Conv2D(256, (5, 5), activation='relu', padding="same"),
            MaxPooling2D((3, 3), strides=(2, 2)),
            # 第三层卷积
            Conv2D(384, (3, 3), activation='relu', padding="same"),
            # 第四层卷积
            Conv2D(384, (3, 3), activation='relu', padding="same"),
            # 第五层卷积
            Conv2D(256, (3, 3), activation='relu', padding="same"),
            MaxPooling2D((3, 3), strides=(2, 2)),
            Flatten(),
            # 全连接层
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        return x