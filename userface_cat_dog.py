import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk

import os
import cv2 as cv
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

height = 64
width = 64
channel = 3

train_data = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1. / 255)

train_set = train_data.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_data.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

class_names = ['cat', 'dog']


def open_image(initialdir='/'):
    file_path = askopenfilename(initialdir=initialdir, filetypes=[('Image File', '*.jpg')])
    img_var.set(file_path)

    image = Image.open(file_path)
    image = image.resize((320, 180))
    photo = ImageTk.PhotoImage(image)
    img_label = Label(middle_frame, image=photo, padx=10, pady=10)
    img_label.image = photo  # keep a reference!
    img_label.grid(row=3, column=1)

    return file_path


def test_image():
    path = img_entry.get()
    if channel == 1:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(path)
    img = cv.resize(img, (height, width))  # resize image to 32x32
    img = img.reshape(1, height, width, channel).astype('float32')
    classifier = models.load_model("model_categorical_complex.model")
    prediction = classifier.predict(img / 255)
    index = np.argmax(prediction)
    # print("Image Loaded")
    # print("Prediction is", class_names[index])
    test_result_var.set(class_names[index])
    return

# ####################################### USER INTERFACE ####################################### #


# creating main application window
root = tk.Tk()
root.geometry("720x720")  # size of the top_frame
root.title("Image Classifier")

# Frame #
top_frame = Frame(root, bd=10)
top_frame.pack()

middle_frame = Frame(root, bd=10)
middle_frame.pack()

bottom_frame = Frame(root, bd=10)
bottom_frame.pack()


# Image input (Top frame)
btn_img_open = Button(top_frame, text='Browse Image',  command=lambda: open_image(img_entry.get()), bg="black",
                      fg="white")
btn_img_open.grid(row=7, column=1)

img_var = StringVar()
img_var.set("/")
img_entry = Entry(top_frame, textvariable=img_var, width=60)
img_entry.grid(row=7, column=2)

btn_img_confirm = Button(top_frame, text='Test Image',  command=test_image, bg="black", fg="white")
btn_img_confirm.grid(row=7, column=4)


test_result_var = StringVar()
test_result_var.set("Your result shown here")
test_result_label = Label(bottom_frame, font=("Courier", 20), height=3, textvariable=test_result_var, bg="white",
                          fg="purple").pack()


# Entering the event mainloop
top_frame.mainloop()


