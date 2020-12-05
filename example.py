#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from frontier_stitching import gen_adversaries, verify


# In[2]:


def to_float(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

def comp(model):
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=["sparse_categorical_accuracy"])

dataset = tfds.load("mnist", split="train", as_supervised=True)
val_set = tfds.load("mnist", split="test", as_supervised=True)

dataset = dataset.map(to_float).shuffle(2048).batch(128).prefetch(-1)
val_set = val_set.map(to_float).batch(128)

model = keras.Sequential([
            keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
            keras.layers.Conv2D(32, 3, padding="same", strides=2, activation="relu"),
            keras.layers.Conv2D(64, 3, padding="same", strides=2, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation=None)
        ])

comp(model)
model.build(input_shape=(None, 28, 28, 1))

l = 100

# generate key set
true_advs, false_advs = gen_adversaries(model, l, dataset, 1. / 255.)

# In case that not the full number of adversaries could be generated a reduced amount is returned
assert(len(true_advs + false_advs) == l)

key_set_x = tf.data.Dataset.from_tensor_slices([x for x, y in true_advs + false_advs])
key_set_y = tf.data.Dataset.from_tensor_slices([y for x, y in true_advs + false_advs])
key_set = tf.data.Dataset.zip((key_set_x, key_set_y)).batch(128)

_ = model.fit(dataset, epochs=3, validation_data=val_set)


# In[3]:


# reset the optimizer and embed the watermark
comp(model)
_ = model.fit(key_set, epochs=2, validation_data=val_set)


# In[4]:


info = verify(model, key_set, 0.05)


# In[5]:


if info["success"]:
    print("Model is ours and was successfully watermarked.")
else:
    print("Model is not ours and was not successfully watermarked.")

