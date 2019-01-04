from flask import Flask, jsonify, request, send_file
from modules.detector import Detector
from keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img, ImageDataGenerator
from keras.models import load_model as load_model_from_file
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append("..")
from utils.models import get_base_model
from constants import img_size
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

pretrained_models = [
    {
        'constructor': ResNet50,
        'preprocessor': resnet50_preprocess_input,
    },
    {
        'constructor': InceptionV3,
        'preprocessor': inception_preprocess_input,
    },
    {
        'constructor': Xception,
        'preprocessor': xception_preprocess_input,
    }
]

image_dir = './images'
detection_dir = 'detection'

detector = None
base_models = None
model = None
classnames = None

def load_detector():
    global detector
    if detector == None:
        detector = Detector('yolo')
    return detector

def load_base_models():
    global base_models
    if base_models is None:
        base_models = []
        for model in pretrained_models:
            base_models.append({
                'model': get_base_model(model['constructor'], img_size=img_size, pooling='avg', verbose=False),
                'preprocessor': model['preprocessor'],
            })
    return base_models

def load_model():
    global model
    global classnames
    if model == None:
        model = load_model_from_file('../input/model/car-identification.h5')
    if classnames is None:
        classnames = np.load('../input/model/classnames.npy')
    return model, classnames

def detect(input_image):
    detector = load_detector()

    filename = secure_filename('image.jpg')
    dir_path = os.path.join(image_dir, detection_dir)
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(image_dir) == False:
        os.mkdir(image_dir)
    if os.path.exists(os.path.join(image_dir, detection_dir)) == False:
        os.mkdir(dir_path)

    input_image.save(filepath)
    
    car_image = detector.crop_image(
        source_image=filepath,
        input_type='file',
        params={
            'detect_largest_box': True,
            'smallest_detected_area': 0.2,
        }
    )

    car_image.save(filepath)

def get_base_features(filepath):
    base_models = load_base_models()

    datagen = ImageDataGenerator(preprocessing_function=resnet50_preprocess_input)

    features = []
    for base_model in base_models:
        datagen = ImageDataGenerator(preprocessing_function=base_model['preprocessor'])
        generator = datagen.flow_from_directory(
            filepath,
            target_size=(img_size, img_size),
            batch_size=1,
            class_mode='categorical',
            shuffle=False,
            seed=0,
        )

        base_model_features = base_model['model'].predict_generator(generator, len(generator), verbose=0)
        print(base_model_features)
        features.append(base_model_features)

    return features

def preprocess_features(features, steps):
    return np.repeat(np.concatenate(features, axis=1), repeats=steps, axis=0)

@app.route('/', methods=["POST"])
def evaluate():
    input_image = request.files.get('image')
    detect(input_image)
    base_features = get_base_features(image_dir)
    features = preprocess_features(base_features, 3)
    features = np.expand_dims(features, axis=0)
    print(features.shape)
    model, classnames = load_model()
    predictions = model.predict(features)
    predictions = predictions[0][3]
    probs = np.flip(np.sort(predictions))[0:5]
    predicted_classes = np.flip(np.argsort(predictions))[0:5]
    predicted_classnames = classnames[predicted_classes]

    return json.dumps([probs.tolist(), predicted_classes.tolist(), predicted_classnames.tolist()])

if __name__ == "__main__":
    print('* Loading detection model...')
    load_detector()
    print('* Loading base models...')
    load_base_models()
    print('* Loading model...')
    load_model()
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)