import logging
import os

import cv2
from flask import Flask
from transformers import YolosForObjectDetection, YolosImageProcessor

from app.services.image_masking import ImageMasker

# Multiple cascade classifiers for face detection in different orientations.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(current_dir)
xml_files = [
    os.path.join(project_root_dir, 'data', 'haarcascade_mcs_leftear.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_mcs_rightear.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_frontalface_default.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_profileface.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_eye.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_upperbody.xml'),
    os.path.join(project_root_dir, 'data', 'haarcascade_mcs_upperbody.xml')
]
cascade_classifiers = [cv2.CascadeClassifier(xml) for xml in xml_files]

# AI model for detecting areas in images.
detection_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


def create_app():
    app = Flask(__name__)

    from app.routes.main import main_blueprint
    from app.routes.masking import masking_blueprint
    app.register_blueprint(main_blueprint)
    app.register_blueprint(masking_blueprint)

    create_logger(app, enable_file_logging=False, enable_console_logging=True)

    return app


def create_logger(app, enable_file_logging: bool = False, enable_console_logging: bool = True):
    app.logger.setLevel(logging.INFO)

    if enable_file_logging:
        file_handler = logging.FileHandler("app.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        app.logger.addHandler(file_handler)

    if enable_console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        app.logger.addHandler(console_handler)
        logging.getLogger().setLevel(logging.INFO)
