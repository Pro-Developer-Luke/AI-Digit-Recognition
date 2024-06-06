#Dependency modules
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

#Importing data
mnist = tf.keras.datasets.mnist #Imports dataset of handwritten digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1) #scales data down from 0-255 to 0-1 
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Neural network model creation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #flattens image to a readable 'Line' shape of 784 px
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #softmax activation determines confidence

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Training the model
model.fit(x_train, y_train, epochs=3)

model.save('trained.keras') #saves the trained model

loss, accuracy = model.evaluate(x_train, y_train)


modelTrained = tf.keras.models.load_model('trained.keras') #Imports training data to determine hand writen digits

decoded_values = []
image_number = 1
while os.path.isfile(f"Samples/digit{image_number}.png"):
    try:
        img = cv.imread(f"Samples/digit{image_number}.png")[:,:,0] #reads sample digit image
        img = np.invert(np.array([img])) #Inverts image
        prediction = modelTrained.predict(img) #Creates a prediction based on modelTrained (epochs data) and mist dataset
        print(f"Sample number [{image_number}]: Prediction is a {np.argmax(prediction)}") #Prints current image data
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show() #Shows matplotlib data of current image
        decoded_values.insert(0, f"{np.argmax(prediction)}") #Adds new prediction to array of predictions from index 0 
    except:
        print("error")
    finally:
        image_number += 1


rounded_accuracy = format(accuracy*100, ".2f") #Format calculation that displays accuracy as a percentage to 2 dps
print(f"\n\nDecoding accuracy:\n{rounded_accuracy}%\n\nEach decoded value:\n{decoded_values}\n\nScript finished.")#Shows epochs accuracy and each prediction that was stored in decoded_values 