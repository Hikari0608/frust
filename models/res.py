import tensorflow as tf

from tensorflow.python.keras import Model
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *

class ResNetBlock(Layer):
    def __init__(self, num_filters, kernel_size, strides, activation='relu', **kwargs):
        super(ResNetBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation1 = Activation(activation) if activation else None
        self.conv2 = Conv2D(num_filters, kernel_size=kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = Add()
        self.activation2 = Activation(activation) if activation else None
    
    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        if self.activation1:
            x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.add([x, inputs])
        if self.activation2:
            x = self.activation2(x)
        return x

class ResNet50(Model):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__(**kwargs)
        self.conv1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation1 = Activation('relu')
        self.max_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')
        
        self.block = Sequential([
            ResNetBlock(64, 3, 1),
            ResNetBlock(64, 3, 1),
            ResNetBlock(64, 3, 1),

            ResNetBlock(128, 3, 1),
            ResNetBlock(128, 3, 1),
            ResNetBlock(128, 3, 1),
            ResNetBlock(128, 3, 1),

            ResNetBlock(256, 3, 1),
            ResNetBlock(256, 3, 1),
            ResNetBlock(256, 3, 1),
            ResNetBlock(256, 3, 1),
            ResNetBlock(256, 3, 1),
            ResNetBlock(256, 3, 1),

            ResNetBlock(512, 3, 1),
            ResNetBlock(512, 3, 1),
            ResNetBlock(512, 3, 1)
        ])
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation='softmax')
    
    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.max_pool(x)
        
        x = self.block(x)
        
        x = self.avg_pool(x)
        x = self.fc(x)
        
        return x

