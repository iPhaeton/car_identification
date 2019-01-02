from flask import Flask, jsonify, request, send_file
from modules.detector import Detector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, save_img
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

detector = None

def load_detector():
    global detector
    if detector == None:
        detector = Detector('yolo')
    return detector

# The request method is POST (this method enables your to send arbitrary data to the endpoint in the request body, including images, JSON, encoded-data, etc.)
@app.route('/', methods=["POST"])
def evaluate():
    input_image = request.files.get('image')
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
    
    return send_file(output_filepath, mimetype='image/jpg')

# The following is for running command `python app.py` in local development, not required for serving on FloydHub.
if __name__ == "__main__":
    print('* Loading detection model...')
    load_detector()
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)