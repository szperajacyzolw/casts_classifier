'''Tekne Consulting blogpost --- teknecons.com'''
'''picture data source: https://www.kaggle.com/ravirajsinh45/real-life-industrial-dataset-of-casting-product'''


from einops import reduce
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import PIL

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    model = models.load_model(os.path.join(this_dir, 'saved_models/model300_0.h5'))
    cast = PIL.Image.open(
        '/tf/notebooks/casts/casting_data/test/0/cast_def_0_1239.jpeg')
    cast_tensor = np.asarray(cast).astype('float32') / 255.
    cast_tensor = np.expand_dims(cast_tensor, axis=0)
    cast_tensor = reduce(
        cast_tensor, 'b h w (c1 c2) -> b h w c1', 'sum', c2=3)

    layer_outs = [layer.output for layer in model.layers[:8]]
    act_model = models.Model(inputs=model.input, outputs=layer_outs)
    activations = act_model.predict(cast_tensor)

    layer_names = [layer for layer in model.layers[:7]]
    img_per_row = 8
    for i, (l_name, l_act) in enumerate(zip(layer_names, activations)):
        n_feat = l_act.shape[-1]
        size0 = l_act.shape[1]
        size1 = l_act.shape[2]
        n_cols = n_feat // img_per_row
        disp_grid = np.zeros((size0 * n_cols, img_per_row * size1))

        for col in range(n_cols):
            for row in range(img_per_row):
                chan_img = l_act[0, :, :, col * img_per_row + row]
                chan_img -= chan_img.mean()
                chan_img /= chan_img.std()
                chan_img *= 64
                chan_img += 128
                chan_img = np.clip(chan_img, 0, 255).astype('uint8')
                disp_grid[col * size0: (col + 1) * size0,
                          row * size1: (row + 1) * size1] = chan_img
        plt.figure(figsize=(75, 45))
        plt.title(str(l_name)[19:-18])
        plt.grid(False)
        plt.axis('off')
        plt.imshow(disp_grid, aspect='equal', cmap='gray')
        plt.savefig(f'layer_{i+1}_def.png', bbox_inches='tight')

    plt.show()
