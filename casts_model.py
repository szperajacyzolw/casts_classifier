'''Tekne Consulting blogpost --- teknecons.com'''
'''picture data source: https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product'''


from tensorflow.keras import mixed_precision
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn import preprocessing as pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow import keras
from einops import reduce
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


''' GPU config'''


config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)
mixed_precision.set_global_policy('mixed_float16')


print(tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))


'''creating data sources'''


def data_generators(train_dir: 'training pictures', test_dir: 'testing pictures',
                    target_size: 'tuple'):
    train_gen = ImageDataGenerator(
        rescale=1. / 255, fill_mode='nearest', validation_split=0.2)  # data already augumented
    test_gen = ImageDataGenerator(rescale=1. / 255)
    train_flow = train_gen.flow_from_directory(train_dir, color_mode='grayscale', class_mode='binary', seed=42,
                                               shuffle=True, batch_size=10, target_size=target_size, subset='training')
    val_flow = train_gen.flow_from_directory(train_dir, color_mode='grayscale', class_mode='binary', seed=421,
                                             batch_size=10, target_size=target_size, subset='validation')
    test_flow = test_gen.flow_from_directory(test_dir, color_mode='grayscale', class_mode='binary', seed=422,
                                             shuffle=True, batch_size=10, target_size=target_size)
    return(train_flow, val_flow, test_flow)


'''simple sequential model'''


def model_init():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(*target_size, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            padding='same', kernel_regularizer='l2'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy',
                                                                            tf.keras.metrics.Precision(),
                                                                            tf.keras.metrics.Recall(),
                                                                            tf.keras.metrics.AUC()])
    return model


def model_name(fold_no):
    return base_savename + str(fold_no) + '.h5'


def model_exec(data_generators):
    train_flow, val_flow, _ = data_generators
    model = model_init()
    checkpoint = keras.callbacks.ModelCheckpoint('saved_models/' + model_name(0),
                                                 monitor='val_accuracy', verbose=2,
                                                 save_best_only=True, mode='max')
    history = model.fit(train_flow, steps_per_epoch=train_flow.samples / train_flow.batch_size,
                        epochs=20, validation_data=val_flow, validation_steps=val_flow.samples / val_flow.batch_size,
                        verbose=2, callbacks=[checkpoint])
    return history


def history_plot(history):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.legend(['train_loss', 'validation_loss'], loc='upper left')
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.legend(['train_accuracy', 'validation_accuracy'], loc='upper left')
    return fig


def model_eval(data_generators):
    _, _, test_flow = data_generators
    model = models.load_model("saved_models/" + base_savename + "0.h5")
    return model.evaluate(test_flow, return_dict=True)


if __name__ == '__main__':
    base_savename = 'model300_'
    target_size = (300, 300)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(this_dir, 'casting_data/train')
    test_dir = os.path.join(this_dir, 'casting_data/test')

    datagen = data_generators(train_dir, test_dir, target_size)
    training_history = model_exec(datagen)
    plot = history_plot(training_history)
    plt.show()

    evaluate = model_eval(datagen)
    print(evaluate)
