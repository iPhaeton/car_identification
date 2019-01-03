from flask import Flask, jsonify, request, send_file
from modules.detector import Detector
from keras.preprocessing.image import img_to_array, load_img, array_to_img, save_img
from werkzeug.utils import secure_filename
import os
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
import sys
sys.path.append("..")
from utils.models import get_base_model
from constants import img_size
import numpy as np
from PIL import Image

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

detector = None
base_models = None

def load_detector():
    global detector
    if detector == None:
        detector = Detector('yolo')
    return detector

def load_base_models():
    global base_models
    if base_models == None:
        base_models = []
        for model in pretrained_models:
            base_models.append({
                'model': get_base_model(model['constructor'], img_size=img_size, pooling='avg', verbose=False),
                'preprocessor': model['preprocessor'],
            })
    return base_models

def detect(input_image):
    detector = load_detector()

    filename = secure_filename('recognizing_image.jpg')
    input_filepath = os.path.join(image_dir, filename)
    
    if os.path.exists(image_dir) == False:
        os.mkdir(image_dir)

    input_image.save(input_filepath)
    
    car_image = detector.crop_image(
        source_image=input_filepath,
        input_type='file',
        params={
            'detect_largest_box': True,
            'smallest_detected_area': 0.2,
        }
    )

    return car_image

def get_base_features(input_image):
    base_models = load_base_models()
    x = img_to_array(input_image.resize((img_size, img_size), Image.NEAREST))
    x = np.expand_dims(x, axis=0)

    features = []
    for base_model in base_models:
        input_x = base_model['preprocessor'](x)
        features.append(base_model['model'].predict(input_x))

    return features

def preprocess_features(features, steps):
    return np.repeat(np.concatenate(features, axis=1), repeats=steps, axis=0)

@app.route('/', methods=["POST"])
def evaluate():
    input_image = request.files.get('image')
    car_image = detect(input_image)
    base_features = get_base_features(car_image)
    
    features = preprocess_features(base_features, 3)
    print(features.shape)

    output_filepath = os.path.join(image_dir, 'recognized_image.jpg')
    save_img(output_filepath, car_image)
    
    return send_file(output_filepath, mimetype='image/jpg')

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print('* Loading detection model...')
    load_detector()
    print('* Loading base models...')
    load_base_models()
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)