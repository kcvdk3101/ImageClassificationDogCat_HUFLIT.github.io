# Image Classification

# Import libraries

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initializing the neural network
classifier = Sequential()

# Add convolution layer
classifier.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add convolution layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add convolution layer
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Add max pooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Add Flatten
classifier.add(Flatten())

# Add full connection
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(units=2, activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Fit CNN to images
# from keras.preprocessing.image import ImageDataGenerator
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

classifier.fit_generator(
    train_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=2000)

classifier.save('model_categorical_complex.model')


# Test accuracy of classifier
def test_accuracy(classifier, test_set, steps):
    num_correct = 0
    num_guesses = 0
    for i in range(steps):
        a = test_set.next()
        guesses = classifier.predict(a[0])
        correct = a[1]
        for index in range(len(guesses)):
            num_guesses += 1
            if round(guesses[index][0]) == correct[index]:
                num_correct += 1
    return num_correct, num_guesses
