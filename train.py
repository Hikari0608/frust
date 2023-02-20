import datasets
import models
import tensorflow as tf
import numpy as np

from tensorflow._api.v2 import io

from models import VGG

model = VGG(weights='imagenet', include_top=True)

# 使用预训练模型进行预测
img = load_img('example.jpg', target_size=(224, 224))
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)
predictions = model.predict(img)