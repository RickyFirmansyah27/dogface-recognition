import os
import glob
import pandas as pd
import numpy as np
import streamlit as st
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping

def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

def validate():
    train_dir = "dataset/split_dataset/train"
    val_dir = "dataset/split_dataset/validation"
    test_dir = "dataset/split_dataset/test"

    train_samples = get_files(train_dir)
    num_classes = len(glob.glob(train_dir + "/*"))
    val_samples = get_files(val_dir)
    test_samples = get_files(test_dir)

    st.write('train_samples:', train_samples)
    st.write('num_classes:', num_classes)
    st.write('val_samples:', val_samples)
    st.write('test_samples:', test_samples)


def process_and_predict(file):
    model = keras.models.load_model("model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    im = file
    width, height = im.size
    if width == height:
        im = im.resize((150, 150), Image.ANTIALIAS)
    else:
        if width > height:
            left = width / 2 - height / 2
            right = width / 2 + height / 2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((150, 150), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((150, 150), Image.ANTIALIAS)
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 150, 150, 3)

    maxnum = np.argmax(model.predict(ar))
    if maxnum == 0:
        prediction = 'Golden'
    elif maxnum == 1:
        prediction = 'Husky'
    elif maxnum != 0 and maxnum != 1:
        prediction = 'Unknown'

    st.write(' is a ' + prediction)

    st.image(im, caption=prediction, use_column_width=True)

if __name__ == "__main__":
    st.sidebar.subheader('Data Testing')
    image = Image.open("test.jpeg")
    file = st.sidebar.file_uploader(label='Pilih data Testing', type=('JPEG'))
      
    if file is not None:
        st.sidebar.write('File Uploaded')
        try:
            image = Image.open(file)
        except Exception as e:
            print(e)
            image = Image.open("test.jpeg")
    
    st.header("Face Recognition using Keras")
   
    if st.button('Testing dengan Keras', key=3):
        st.write('Sedang melakukan testing')
        validate()
        if image is not None:
            process_and_predict(image)
        else:
            st.write('File testing belum diunggah')
    else:
        st.write('')
