from flask import Flask, jsonify, request, send_file, make_response
from modules.detector import Detector
from keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img, ImageDataGenerator
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append("..")
from utils.models import get_base_model
from utils.response import create_response
from utils.image import get_image_of_size
from constants import img_size
import numpy as np
from PIL import Image
import json
import pandas
import base64
import re

pretrained_models = [
    {
        'path': '../input/base-models/resnet50.json',
        'weights': '../input/base-models/resnet50.h5',
        'preprocessor': resnet50_preprocess_input,
    },
    {
        'path': '../input/base-models/inception_v3.json',
        'weights': '../input/base-models/inception_v3.h5',
        'preprocessor': inception_preprocess_input,
    },
    
    {
        'path': '../input/base-models/xception.json',
        'weights': '../input/base-models/xception.h5',
        'preprocessor': xception_preprocess_input,
    }
]

image_dir = './images'
detection_dir = 'detection'
preview_dir = '../input/preview'

detector = None
base_models = None
model = None
classnames = None
preview_data = pandas.read_csv('../input/preview/preview_data.csv')

def load_detector():
    global detector
    if detector == None:
        detector = Detector('yolo')
    return detector

def load_model_from_file(path, weights):
    with open(path, 'r') as inputfile:
        model_json = json.load(inputfile)
    model = model_from_json(model_json)
    model.load_weights(weights)
    return model

def load_base_models():
    global base_models
    if base_models is None:
        base_models = []
        for model in pretrained_models:
            base_models.append({
                'model': load_model_from_file(model['path'], model['weights']),
                'preprocessor': model['preprocessor'],
            })
    return base_models

def load_model():
    global model
    global classnames
    if model == None:
        model = load_model_from_file('../input/model/car-identification.json', '../input/model/car-identification.h5')
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
        features.append(base_model_features)

    return features

def preprocess_features(features, steps):
    return np.repeat(np.concatenate(features, axis=1), repeats=steps, axis=0)


app = Flask(__name__)
# print('* Loading detection model...')
# load_detector()
# print('* Loading base models...')
# load_base_models()
# print('* Loading model...')
# load_model()
# print('App ready')

@app.route('/', methods=["POST"])
def evaluate():
    print('Evaluating...')
    input_image = request.files.get('image')
    
    limit = request.form.get('limit')
    if limit == None:
        limit = 5
    else:
        limit = int(limit)

    detect(input_image)
    base_features = get_base_features(image_dir)
    features = preprocess_features(base_features, 3)
    features = np.expand_dims(features, axis=0)
    model, classnames = load_model()
    predictions = model.predict(features)
    predictions = predictions[0][3]
    probs = np.flip(np.sort(predictions))[0:limit]
    predicted_classes = np.flip(np.argsort(predictions))[0:limit]
    predicted_classnames = classnames[predicted_classes]

    return json.dumps([probs.tolist(), predicted_classes.tolist(), predicted_classnames.tolist()])

@app.route('/preview', methods=['GET'])
def get_preview():
    filename = request.args.get('filename')
    five_top_names = None
    five_top_probs = None

    if filename == None:
        index = np.random.randint(0, preview_data.shape[0])
        filename = preview_data['filename'][index]
        five_top_names = preview_data['five_top_names'][index]
        five_top_names = re.sub('[!\n]', '', five_top_names)[2:-2].split("' '")
        five_top_probs = preview_data['five_top_probs'][index]
        five_top_probs = re.sub('[!\n]', '', five_top_probs)[2:-1].split("  ")
        five_top_probs = list(map(lambda x: float(x), five_top_probs))
    
    file_path = os.path.join(preview_dir, filename)
    
    response = 'Something went wrong'
    if (filename != None) & (os.path.exists(file_path) == False):
        response = create_response(json.dumps({'error': f'No file "{file_path}" among previews'}), 400)
    else:
        image_base64 = get_image_of_size(file_path, [800, 500], True)
        response = create_response(json.dumps({'image_base64': image_base64, 'classes': five_top_names, 'probs': five_top_probs}))
        
    return response

@app.route('/thumbnails', methods=['GET'])
def get_thumbnails():
    quantity = request.args.get('quantity')
    
    if quantity == None:
        response = create_response(json.dumps({'error': 'Parameter "quantity" is required'}), 400)
        return response

    indices = np.random.choice(preview_data.shape[0], int(quantity))
    
    thumbnails = []
    for index in indices:
        filename = preview_data['filename'][index]
        file_path = os.path.join(preview_dir, filename)
        image_base64 = get_image_of_size(file_path, [200, 200], False)
        thumbnails.append({'image_base64': image_base64, 'filename': filename})
    
    response = create_response(json.dumps(thumbnails))
    return response

if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False, debug=True)