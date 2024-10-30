import os
import cv2
import pandas as pd
from measurements.new_measurements import FaceMeasurements
from face_landmarks import FaceLandmarks, draw_landmarks_on_image


def _facial_landmarks(image_path):
    
    face_landmarks = FaceLandmarks(image_path)
    image = draw_landmarks_on_image(cv2.imread(image_path), 
                                           face_landmarks.get_all_landmarks())
    
    return image


def _get_measurements(image_path):
    """
    Extract all face measurements from the given image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing all extracted measurements.
    """
    # Extract landmarks using the wrapper function
    image_with_landmark = _facial_landmarks(image_path)
    
    # Initialize FaceMeasurements with the extracted landmarks
    face_measurements = FaceMeasurements(image_path)
    
    # Get all measurements
    measurements = face_measurements.get_all_measurements()
    return measurements, image_with_landmark


def get_face_measurements(image_path):
    measurements, image_with_landmark = _get_measurements(image_path)

    measurements = {
        'Face Rectangularity': measurements.get('face_rectangularity'),
        'Middle Face Rectangularity': measurements.get('middle_face_rectangularity'),
        'Forehead Rectangularity': measurements.get('forehead_rectangularity'),
        'Chin Angle': measurements.get('chin_angle'),
        'Ratio Lower to Middle Width': measurements.get('ratio_lower_to_middle_width'),
        'Ratio Upper to Middle Width': measurements.get('ratio_upper_to_middle_width'),
        'Difference Ratio Upper Lower': measurements.get('difference_ratio_upper_lower'),
        'Face Aspect Ratio': measurements.get('face_aspect_ratio'),
        'Face Ratio': measurements.get('face_ratio'),
        'Forehead Ratio': measurements.get('forehead_ratio'),
        'Cheekbone Ratio': measurements.get('cheekbone_ratio'),
        'Jawline Ratio': measurements.get('jawline_ratio'),
        'Chin Ratio': measurements.get('chin_ratio'),
        'Jaw Angle': measurements.get('jaw_angle'),
        'Face Length': measurements.get('face_length'),
        'Face Width': measurements.get('face_width'),
        'Forehead Width': measurements.get('forehead_width'),
        'Jaw Width': measurements.get('jaw_width'),
        'Chin Height': measurements.get('chin_height'),
        'Chin to Lip Height': measurements.get('chin_to_lip_height'),
        'Nose to Hairline Height': measurements.get('nose_to_hairline_height'),
        'Jawline Angularity': measurements.get('jawline_angularity'),
        'Chin Curvature': measurements.get('chin_curvature'),
        'Nose Size': measurements.get('nose_size'),
        'Left Eye Size': measurements.get('left_eye_size'),
        'Right Eye Size': measurements.get('right_eye_size'),
        'Inter Eye Distance': measurements.get('inter_eye_distance'),
        'Cheekbone Angle': measurements.get('cheekbone_angle'),
        'Lip Size': measurements.get('lip_size'),
    }

    image_with_landmark = cv2.cvtColor(image_with_landmark, cv2.COLOR_BGR2RGB)

    return measurements, image_with_landmark
