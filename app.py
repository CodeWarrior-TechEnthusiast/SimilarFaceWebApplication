import os
import sys

from flask import Flask, render_template, url_for, request, Response
from cv2 import cv2
import random
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
from keras.utils.image_utils import img_to_array, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)
app.config['UPLOAD_FOLDER'] = './uploads'
final = 0

class ImageProcessing:
    def __init__(self):
        self.batch_size = 10
        self.num_classes = 5
        self.epochs = 3
    def train(self):
        # VGG16 model without the fully connected layers
        base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        # Adding custom layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        self.model = Model(inputs=base_model.input, outputs=predictions)
        # Compiling the model
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # Define the data generators
        self.train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)
        # Create the training and validation generators
        self.train_generator = self.train_datagen.flow_from_directory(r'C:\Users\advai\Desktop\MySimilarFace\MySimilarFace\uploads',
                                                            target_size=(224, 224),
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical')
        self.history = self.model.fit(self.train_generator,
                                      steps_per_epoch=self.train_generator.samples // self.batch_size,
                                      epochs=self.epochs)
    def predict(self, frame):
        img = load_img(frame, target_size=(224, 224))
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        # Predicting the class
        prediction = self.model.predict(img_tensor)

        # the index of the class with the highest probability
        class_index = np.argmax(prediction[0])

        # Get the class label
        class_label = self.train_generator.class_indices
        class_label = {v: k for k, v in class_label.items()}
        predicted_class = class_label[class_index]

        print("Predicted class: ", predicted_class)
        return predicted_class

model = ImageProcessing()


def proc():
    while True:
        f, frame = camera.read()
        BLACK = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.1
        font_color = BLACK
        font_thickness = 2
        x, y = 50, 50
        cv2.imwrite("temp.jpg", frame)
        frame = cv2.putText(frame, model.predict("temp.jpg"), (x, y), font, font_size, font_color, font_thickness, cv2.LINE_AA)
        a, b = cv2.imencode('.jpg', frame)
        frame = b.tobytes()
        yield b''.join([b'--frame\r\n', b'Content-Type: image/jpeg\r\n\r\n', frame, b'\r\n'])


@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/try/', methods=('GET', 'POST'))
def main_app():
    if request.method == "POST":
        file = request.files.getlist("file")
        #file = request.files['file']
        for x in file:
            x.save(os.path.join(r"C:/Users/advai/Desktop/MySimilarFace/MySimilarFace/uploads/" + x.filename[0], x.filename))
        model.train()
        return render_template('results.html')
    return render_template('try.html')

@app.route('/video')
def video():
    return Response(proc(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()

