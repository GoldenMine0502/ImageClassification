import os

import keras.optimizer_v2.adam
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tfimm
import pathlib

# 파일 위치들
train_data_dir = "./images3"
# train_data_dir = "./images2"
# test_data_dir = "./bmd_dataset/test"

# 매개변수 정의
batch_size = 256
img_height = 54
img_width = 54
epochs = 100
validation_split = 0.05
filter_multiplier = 1
output_file_name = 'write.csv'

# train 데이터셋 적용
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="training",
    seed=1234,
    # color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

# validation 데이터셋 적용
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    validation_split=validation_split,
    subset="validation",
    seed=1234,
    # color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

# class_names = train_ds.class_names
# print(class_names)

num_classes = len(set(os.listdir(train_data_dir)))

# tfimm.create_model()


model = Sequential()


# 쓰레기...
# model.add(layers.experimental.preprocessing.RandomFlip(input_shape=(img_height, img_width, 3)))
# model.add(layers.experimental.preprocessing.RandomRotation(0.01, input_shape=(img_height, img_width, 3)))
# model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)))
# layers.experimental.preprocessing.RandomRotation(0.1, input_shape=(img_height,
#                                                                    img_width,
#                                                                    1)),
# layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 1)),
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  # input_shape=(img_height, img_width, 3),
                                                  pooling='avg',
                                                  classes=num_classes,
                                                  weights='imagenet'
                                                  )
# pretrained_model = tfimm.create_model("vit_tiny_patch16_224",
#                             # pretrained=True
#                            )

# def get_model(shape=(32, 32, 3), weights="imagenet"):
#     model = tf.keras.applications.ResNet50(input_shape=shape, include_top=False, weights=weights)
#     flatten = tf.keras.layers.GlobalAveragePooling2D()(model.output)
#     drop_out = tf.keras.layers.Dropout(0.5)(flatten)
#     dense = tf.keras.layers.Dense(2048, activation="relu")(drop_out)
#     prediction = tf.keras.layers.Dense(num_classes, activation="softmax", name="prediction")(dense)
#     model = tf.keras.Model(model.input, prediction)
#     return model
#
#
# pretrained_model = get_model(shape=(img_height, img_width, 3))

# pretrained_model = tf.keras.applications.resnet.ResNet152(
#     include_top=False,
#     weights='imagenet',
#     # input_tensor=None,
#     input_shape=(img_height, img_width, 3),
#     # pooling='avg',
#     classes=num_classes,
# )
# # for layer in pretrained_model.layers:
# #     layer.trainable = False
#
# # pretrained_model.layers[0].trainable = False
#
model.add(pretrained_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# model.load_weights("./model/6.8870-09.hdf5")

# model = Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])

# model = Sequential([
#     layers.experimental.preprocessing.RandomFlip(input_shape=(img_height,
#                                                               img_width,
#                                                               3)),
#     layers.experimental.preprocessing.RandomRotation(0.1, input_shape=(img_height,
#                                                                        img_width,
#                                                                        3)),
#     layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
#     layers.ZeroPadding2D(padding=(2, 2)),
#     layers.Conv2D(16 * filter_multiplier,
#                   (3, 3),
#                   activation='relu',
#                   # padding='same',
#                   # bias_regularizer=regularizers.l2(1e-5),
#                   # kernel_regularizer=regularizers.l2(1e-5)
#                   ),
#     layers.Conv2D(16 * filter_multiplier,
#                   (3, 3),
#                   activation='relu',
#                   # bias_regularizer=regularizers.l2(1e-5),
#                   # kernel_regularizer=regularizers.l2(1e-5)
#                   ),
#     layers.MaxPooling2D(),
#     # layers.GaussianNoise(0.1),
#     # layers.Dense(1),
#
#     layers.SeparableConv2D(32 * filter_multiplier, (3, 3), activation='relu'),
#     layers.SeparableConv2D(32 * filter_multiplier, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),
#
#     layers.SeparableConv2D(64 * filter_multiplier, (3, 3), activation='relu'),
#     layers.SeparableConv2D(64 * filter_multiplier, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),
#
#     layers.SeparableConv2D(128 * filter_multiplier, (3, 3), activation='relu'),
#     layers.SeparableConv2D(128 * filter_multiplier, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),
#
#     layers.SeparableConv2D(256 * filter_multiplier, (3, 3), activation='relu'),
#     layers.SeparableConv2D(256 * filter_multiplier, (3, 3), activation='relu'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(),
#     layers.Dropout(0.25),
#
#     # layers.SeparableConv2D(256, (3, 3), activation='relu'),
#     # layers.SeparableConv2D(256, (3, 3), activation='relu'),
#     # layers.BatchNormalization(),
#     # layers.MaxPooling2D(),
#     # layers.Dropout(0.5),
#
#     layers.Flatten(),
#
#     layers.Dense(512 * filter_multiplier, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.5),
#
#     layers.Dense(128 * filter_multiplier, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.25),
#
#     layers.Dense(32 * filter_multiplier, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.25),
#
#     # layers.Dense(16, activation='relu'),
#     # layers.BatchNormalization(),
#     # layers.Dropout(0.1),
#
#     layers.Dense(num_classes)
# ])
#


model.summary()

model.compile(optimizer=Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model_path = './model/' + '{val_loss:.4f}-{epoch:02d}.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[cb_checkpoint]
    # callbacks=[earlyStopping]
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('final.h5')
