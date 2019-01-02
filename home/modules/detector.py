from imageai.Detection import ObjectDetection
import os
import sys
sys.path.append("..")
from utils.files import filter_files
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img, save_img
import numpy as np

class Detector:
    def __init__(self, model):
        execution_path = os.getcwd()
        detector = ObjectDetection()

        if model == 'yolo':
            detector.setModelTypeAsYOLOv3()
        elif model == 'yolo-tiny':
            detector.setModelTypeAsTinyYOLOv3()
        else:
            raise ValueError('Model ' + model + 'not fould. you should download the model and put it into "modules" directory.')

        detector.setModelPath(os.path.join(execution_path , './modules/' + model + '.h5'))
        detector.loadModel()
        custom_objects = detector.CustomObjects(car=True)

        self.detector = detector
        self.custom_objects = custom_objects
        self.execution_path = execution_path

    def get_box(self, detections, size, params = {'detect_largest_box': False, 'smallest_detected_area': None}):
        detect_largest_box = params['detect_largest_box']
        smallest_detected_area = params['smallest_detected_area']
                
        area = 0
        points = None

        if detect_largest_box == True:
            for detection in detections:
                current_area = (detection['box_points'][2] - detection['box_points'][0]) * (detection['box_points'][3] - detection['box_points'][1])
                if current_area > area:
                    area = current_area
                    points = detection['box_points']
        else:
            points = detections[0]['box_points']

        img_area = size[0] * size[1] if smallest_detected_area != None else None
        
        if (img_area != None) and (area / img_area < smallest_detected_area):
            return None
        else:
            return points

    def crop_image(self, source_image, input_type, **kwargs):
        image = None
        input_image = None
        if input_type == 'file':
            image = load_img(source_image)
            input_image = source_image
        elif input_type == 'array':
            image = source_image
            input_image = img_to_array(source_image)

        detections = self.detector.detectCustomObjectsFromImage(
            custom_objects=self.custom_objects,
            input_type=input_type,
            input_image=input_image, 
            minimum_percentage_probability=10,
            extract_detected_objects=False,
        )

        box = self.get_box(detections, image.size, **kwargs)
        cropped_img = image.crop(box)

        return cropped_img

    def crop_dir(self, source_path, target_path, directory, **kwargs):
        if os.path.exists(target_path + directory) == False:
            os.mkdir(target_path + directory)
        
        filenames = os.listdir(source_path + directory)
        filenames = filter_files(filenames)
        
        for filename in filenames:
            image_path = os.path.join(self.execution_path , source_path + directory + '/' + filename)
            cropped_img = self.crop_image(image_path, 'file', **kwargs)
            save_img(target_path + directory + '/' + filename, cropped_img)
        
    def crop_path(self, source_path, target_path, dirs, **kwargs):
        target_dirs = os.listdir(target_path)
        dirs = filter_files(dirs, target_dirs)

        if os.path.exists(target_path) == False:
            os.mkdir(target_path)
        
        for i, directory in enumerate(dirs):
            print('Directory', str(i), 'of', str(len(dirs)), '...')
            self.crop_dir(source_path, target_path, directory, **kwargs)
            print('Images in directory', str(i), '"' + directory + '"', 'cropped')
