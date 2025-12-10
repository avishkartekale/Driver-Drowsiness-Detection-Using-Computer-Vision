# Installation notes (Windows PowerShell):
# - Create and activate a virtual environment (recommended):
#     python -m venv .venv; .\.venv\Scripts\Activate.ps1
# - Install TensorFlow (recommended):
#     pip install --upgrade pip; pip install "tensorflow>=2.10"
# - If you prefer the standalone Keras (not recommended), install:
#     pip install --upgrade pip; pip install keras
# If you get a ModuleNotFoundError: No module named 'keras', it's because
# the environment where you run Python doesn't have TensorFlow/Keras installed.

import os
# Prefer the bundled Keras inside TensorFlow if available (recommended).
# Fall back to the standalone `keras` package when TensorFlow is not installed.
try:
    # TensorFlow 2.x exposes Keras as `tensorflow.keras` which is the recommended API.
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
except Exception:
    # If tensorflow is not installed, try the standalone keras package.
    try:
        
        from keras.preprocessing import image
        from keras.utils.np_utils import to_categorical
        from keras.models import Sequential, load_model
        from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
    except Exception as e:
        raise ImportError(
            "Could not import Keras. Install TensorFlow (recommended) with `pip install tensorflow` "
            "or the standalone Keras with `pip install keras`. Original error: {}".format(e)
        )

import matplotlib.pyplot as plt
import numpy as np
import random, shutil


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnnCat2.h5', overwrite=True)