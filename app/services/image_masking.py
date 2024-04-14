import mimetypes
from io import BytesIO
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from cv2.typing import MatLike, Rect, Scalar
from werkzeug.datastructures.file_storage import FileStorage


class ImageMasker:
    """
    A class for masking images by detecting areas containing faces and replacing them with black rectangles.
    """

    def __init__(self, cascade_classifiers: list[cv2.CascadeClassifier], detection_model, image_processor):
        """
        Initializes the ImageMasker with cascade classifiers, a detection model and an image processor.
        """
        self.cascade_classifiers = cascade_classifiers
        self.detection_model = detection_model
        self.image_processor = image_processor

        self.min_aspect = 0.5
        self.max_aspect = 2.0
        self.processing_threshold = 0.2

    def mask_file(self, file: FileStorage,
                  allow_full_mask: bool = False,
                  should_draw_gizmos: bool = False) -> Tuple[bytes, int]:
        """
        Masks the image file by detecting areas with faces and replacing them with black rectangles.
        The image file is read into bytes, and the resulting masked image is returned as bytes.
        :param file: The image file to mask.
        :param allow_full_mask: Whether to allow full masking of the image.
        :param should_draw_gizmos: Whether to draw gizmos around the detected areas.
        :return: The masked image as bytes.
        """
        guessed_extension = mimetypes.guess_extension(file.content_type)
        return self.mask_data(file.read(), guessed_extension, allow_full_mask, should_draw_gizmos)

    def mask_data(self, image_as_bytes: bytes,
                  extension: str = '.png',
                  allow_full_mask: bool = False,
                  should_draw_gizmos: bool = False) -> Tuple[bytes, int]:
        """
        Masks the image data by detecting areas with faces and replacing them with black rectangles.
        The image data is read from bytes, and the resulting masked image is returned as bytes.
        :param image_as_bytes: The image data to mask.
        :param extension: The extension of the image file.
        :param allow_full_mask: Whether to allow full masking of the image.
        :param should_draw_gizmos: Whether to draw gizmos around the detected areas.
        :return: The masked image as bytes.
        """
        mat = cv2.imdecode(np.frombuffer(image_as_bytes, np.uint8), cv2.IMREAD_COLOR)
        image = Image.open(BytesIO(image_as_bytes))
        areas_of_interest = self.find_rects_of_interest(image, allow_full_mask)
        areas, detected_faces = self.detect_maskable_areas(mat, areas_of_interest)
        if should_draw_gizmos:
            self.draw_gizmos(areas_of_interest, mat, color=(0, 0, 0))
            self.draw_gizmos(areas, mat, color=(255, 0, 0))
            self.draw_gizmos(detected_faces, mat, color=(0, 0, 255))
        else:
            self.mask_areas(areas, mat)
        _, encoded_result = cv2.imencode(extension, mat)
        return encoded_result.tobytes(), len(areas)

    def mask_areas(self, areas, image: MatLike) -> None:
        """
        Masks all the specified areas in the image by replacing them with black rectangles.
        :param areas: The areas to mask.
        :param image: The image to mask.
        """
        for area in areas:
            self.mask_area(area, image)

    def mask_area(self, area, image: MatLike) -> None:
        """
        Masks a single area in the image by replacing it with a black rectangle.
        :param area: The area to mask.
        :param image: The image to mask.
        """
        cv2.rectangle(image, area[0:2], (area[0] + area[2], area[1] + area[3]), (0, 0, 0), -1)

    def draw_gizmos(self, areas, image: MatLike, color: Scalar = (255, 255, 0)) -> None:
        """
        Draws gizmos around the specified areas in the image.
        :param areas: The areas to draw gizmos around.
        :param image: The image to draw gizmos on.
        :param color: The color to use for the gizmos.
        """
        for area in areas:
            cv2.rectangle(image, area[0:2], (area[0] + area[2], area[1] + area[3]), color, 2)

    def detect_maskable_areas(self, image: MatLike, areas_of_interest: List[Rect]) -> Tuple[List[Rect], List[Rect]]:
        """
        Detects maskable areas in the image using the cascade classifiers.
        They are checked against the areas of interest and only kept if a face is detected.
        Returns a list of rectangles representing the detected areas.
        :param image: The image to detect maskable areas in.
        :param areas_of_interest: The areas of interest to check against.
        :return: A list of rectangles representing the detected areas.
        """
        areas = []
        detected_gizmos = []
        for rect in areas_of_interest:
            partial_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            detected = self.detect_faces(partial_image)
            if len(detected) > 0:
                areas.append(rect)
                detected = [(d[0] + rect[0], d[1] + rect[1], d[2], d[3]) for d in detected]
                detected_gizmos.extend(detected)
        return areas, detected_gizmos

    def find_rects_of_interest(self, image: Image, allow_full_mask: bool = False) -> List[Rect]:
        """
        Finds rectangles of interest in the image using the AI detection model.
        Returns a list of rectangles representing the areas of interest.
        :param image: The image to find rectangles of interest in.
        :param allow_full_mask: Whether to allow full masking of the image.
        :return: A list of rectangles representing the areas of interest.
        """
        rects = []
        if allow_full_mask:
            rects.append((0, 0, image.size[0], image.size[1]))
            return rects

        image = image.convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.detection_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.processing_threshold,
            target_sizes=target_sizes
        )[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(round(i, 0)) for i in box.tolist()]
            rect = (box[0], box[1], box[2] - box[0], box[3] - box[1])
            if allow_full_mask or not self.is_overlapping_full_image(rect, image):
                rects.append(rect)
        return rects

    def is_within_aspect_ratio(self, rect: Rect) -> bool:
        """
        Checks if the aspect ratio of the rectangle is within the specified range.
        Returns True if the aspect ratio is within range, False otherwise.
        :param rect: The rectangle to check the aspect ratio of.
        :return: True if the aspect ratio is within range, False otherwise.
        """
        aspect_ratio = rect[2] / rect[3]
        return self.min_aspect <= aspect_ratio <= self.max_aspect

    def is_overlapping_full_image(self, rect: Rect, image: Image) -> bool:
        """
        Checks if the rectangle overlaps more than 70% of the image.
        :param rect: The rectangle to check for overlap.
        :param image: The image to check for overlap with.
        :return: True if the rectangle overlaps more than 70% of the image, False otherwise.
        """
        bbox = image.size
        return ((rect[2] / bbox[0]) > 0.7) or ((rect[3] / bbox[1]) > 0.7)

    def detect_faces(self, image: MatLike) -> List[Rect]:
        """
        Detects faces in the image using the cascade classifiers.
        Returns True if at least one face is detected, False otherwise.
        :param image: The image to detect faces in.
        :return: A tuple containing a list of tuples for each detected face.
        """
        rects = []
        for classifier in self.cascade_classifiers:
            faces_detected = classifier.detectMultiScale(image)
            if len(faces_detected) > 0:
                for (x, y, w, h) in faces_detected:
                    rects.append((x, y, w, h))
                return rects
        return rects
