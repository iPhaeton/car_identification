from flask import Flask, jsonify, request, send_file
from modules.detector import Detector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, save_img
from werkzeug.utils import secure_filename
import os
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
import sys
sys.path.append("..")
from utils.models import get_base_model
from constants import img_size

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
            base_models.append(get_base_model(model['constructor'], model['preprocessor'], img_size=img_size, pooling='avg', verbose=False))
    return base_models

def detect(input_image):
    detector = load_detector()

    filename = secure_filename('recognizing_image.jpg')
    image_dir = './images'
    input_filepath = os.path.join(image_dir, filename)
    output_filepath = os.path.join(image_dir, 'recognized_image.jpg')

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

    save_img(output_filepath, car_image)
    car_image_array = img_to_array(car_image)
    return output_filepath

def get_base_features():
    base_models = load_base_models()
    print(len(base_models))

@app.route('/', methods=["POST"])
def evaluate():
    input_image = request.files.get('image')
    output_filepath = detect(input_image)
    get_base_features()
    
    return send_file(output_filepath, mimetype='image/jpg')

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print('* Loading detection model...')
    load_detector()
    print('* Loading base models...')
    load_base_models()
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)