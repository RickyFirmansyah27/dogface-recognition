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
from sklearn.metrics import confusion_matrix

def get_files(directory):
    if not os.path.exists(directory):
        return 0
    count = 0
    for current_path, dirs, files in os.walk(directory):
        for dr in dirs:
            count += len(glob.glob(os.path.join(current_path, dr + "/*")))
    return count

# def validate():
#     model = keras.models.load_model("model.h5")
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     true_labels = []
#     predictions = []
#     train_dir = "dataset/split_dataset/train"
#     val_dir = "dataset/split_dataset/validation"
#     test_dir = "dataset/split_dataset/test"

#     train_samples = get_files(train_dir)
#     num_classes = len(glob.glob(train_dir + "/*"))
#     val_samples = get_files(val_dir)
#     test_samples = get_files(test_dir)

#     st.write('train_samples:', train_samples)
#     st.write('num_classes:', num_classes)
#     st.write('val_samples:', val_samples)
#     st.write('test_samples:', test_samples)

#     true_labels.extend(['Golden'] * val_samples)
#     true_labels.extend(['Husky'] * val_samples)
#     predictions.extend(['Golden'] * val_samples)
#     predictions.extend(['Husky'] * val_samples)

#     st.write('Confusion Matrix:')
#     st.write(confusion_matrix(true_labels, predictions))
def validate():
    model = keras.models.load_model("model.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    true_labels = []
    predictions = []
    val_dir = "dataset/split_dataset/validation"

    val_samples = get_files(val_dir)
    num_classes = model.output_shape[-1]

    # Iterate over the validation directory and predict labels for each image
    for current_path, dirs, files in os.walk(val_dir):
        for file in files:
            image_path = os.path.join(current_path, file)
            im = Image.open(image_path)
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

            # Perform prediction using the model
            predictions.append(np.argmax(model.predict(ar)))
            true_labels.append(os.path.basename(current_path))

    st.write('Confusion Matrix:')
    st.write(confusion_matrix(true_labels, predictions, labels=range(num_classes)))



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
