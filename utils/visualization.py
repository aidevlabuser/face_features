# visualization.py

import cv2
import mediapipe as mp
import numpy as np
import os

# Define colors for visualization
FACE_MESH_COLOR = (0, 255, 0)  # Green for face mesh
DLIB_FOREHEAD_COLOR = (0, 0, 255)  # Red for dlib forehead point
MEASUREMENT_LINE_COLOR = (255, 0, 0)  # Blue for measurement lines

# Initialize Mediapipe's Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def draw_face_mesh(image, landmarks):
    """
    Draw the face mesh using Mediapipe landmarks on the given image.

    Parameters:
        image (numpy.ndarray): The image to draw on.
        landmarks (list): List of (x, y) tuples representing the Mediapipe landmarks.
    """
    for landmark in landmarks:
        cv2.circle(image, landmark, 1, FACE_MESH_COLOR, -1)

def draw_forehead_point(image, point):
    """
    Draw the dlib forehead point on the given image.

    Parameters:
        image (numpy.ndarray): The image to draw on.
        point (tuple): (x, y) tuple representing the dlib forehead point.
    """
    if point:
        cv2.circle(image, point, 3, FACE_MESH_COLOR, -1)

def draw_bounding_rectangle(image, points, name):
    """
    Draw a bounding rectangle around given points.

    Parameters:
        image (numpy array): Image on which to draw.
        points (list): List of tuples representing the coordinates of the points.
        name (str): Name of the rectangularity to display.
    """
    if len(points) < 2:
        return

    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Draw rectangle
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

    # Display name
    cv2.putText(image, name, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


def draw_measurement_line(image, point1, point2, label):
    """
    Draw a line between two points with a label for measurements.
    """
    if point1 is None or point2 is None:
        print(f"Cannot draw line: One of the points is None: {point1}, {point2}")
        return

    if len(point1) != 2 or len(point2) != 2:
        print(f"Cannot draw line: Invalid points: {point1}, {point2}")
        return

    # Draw the line between the points
    cv2.line(image, point1, point2, MEASUREMENT_LINE_COLOR, 2)

    # Optionally add a label or text to indicate the measurement
    midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(image, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def visualize_landmarks(image_path, landmarks_extractor, measurements):
    """
    Visualize the landmarks and measurements on the given image.

    Parameters:
        image_path (str): Path to the image file.
        landmarks_extractor (object): The landmarks extractor object with Mediapipe and dlib properties.
        measurements (dict): Dictionary containing measurements and their corresponding points.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
     # Create a copy of the image for drawing landmarks
    image_landmarks = image.copy()

    # Draw the face mesh using Mediapipe landmarks
    draw_face_mesh(image_landmarks, landmarks_extractor.landmarks)

    # Draw the forehead point using dlib
    draw_forehead_point(image_landmarks, landmarks_extractor.forehead)

     # Show the image with visualizations
    cv2.imshow("Face Measurements Visualization", image_landmarks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    image_measurements = image.copy()

    # Draw the face mesh again for context
    draw_face_mesh(image_measurements, landmarks_extractor.landmarks)
    draw_forehead_point(image_measurements, landmarks_extractor.forehead)

     # Draw measurement lines
    for key, (point1, point2) in measurements.items():
        # Validate points before drawing
        if point1 is None or point2 is None:
            print(f"Error: {key} has invalid points: {point1}, {point2}")
            continue
        
        if len(point1) != 2 or len(point2) != 2:
            print(f"Error: {key} points do not have valid coordinates: {point1}, {point2}")
            continue

        # Draw the line if the points are valid
        draw_measurement_line(image_measurements, point1, point2, key)

    # Show the image with visualizations
    cv2.imshow("Face Measurements Visualization", image_measurements)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_visualization(image_path, landmarks_extractor, measurements, landmarks_output_dir, measurements_output_dir):
    """
    Visualize the landmarks and measurements on the given image and save them.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}.")
        return

    # Create a copy of the image for drawing landmarks
    image_landmarks = image.copy()

    # Draw the face mesh using Mediapipe landmarks
    draw_face_mesh(image_landmarks, landmarks_extractor.landmarks)

    # Draw the forehead point using dlib
    draw_forehead_point(image_landmarks, landmarks_extractor.forehead)

    # Save the image with landmarks
    landmarks_output_path = os.path.join(landmarks_output_dir, os.path.basename(image_path))
    cv2.imwrite(landmarks_output_path, image_landmarks)
    print(f"Landmarks image saved at {landmarks_output_path}")

    # Create a copy of the image for drawing measurements
    image_measurements = image.copy()

    # Draw the face mesh again for context
    draw_face_mesh(image_measurements, landmarks_extractor.landmarks)
    draw_forehead_point(image_measurements, landmarks_extractor.forehead)

    # Draw measurement lines
    for key, (point1, point2) in measurements.items():
        # Validate points before drawing
        if point1 is None or point2 is None:
            print(f"Error: {key} has invalid points: {point1}, {point2}")
            continue
        
        if len(point1) != 2 or len(point2) != 2:
            print(f"Error: {key} points do not have valid coordinates: {point1}, {point2}")
            continue

        # Draw the line if the points are valid
        draw_measurement_line(image_measurements, point1, point2, key)

    # Save the image with measurements
    measurements_output_path = os.path.join(measurements_output_dir, os.path.basename(image_path))
    cv2.imwrite(measurements_output_path, image_measurements)
    print(f"Measurements image saved at {measurements_output_path}")

def new_draw_measurement_line(image, point1, point2):
    """
    Draw a line between two points without any label.

    Parameters:
        image (numpy.ndarray): The image to draw on.
        point1 (tuple): (x, y) of the first point.
        point2 (tuple): (x, y) of the second point.
    """
    # Draw line between the points
    cv2.line(image, point1, point2, MEASUREMENT_LINE_COLOR, 2)


def new_visualize_landmarks(image_path, landmarks_extractor, measurements):
    """
    Visualize the landmarks and measurements on the given image.

    Parameters:
        image_path (str): Path to the image file.
        landmarks_extractor (object): The landmarks extractor object with Mediapipe and dlib properties.
        measurements (dict): Dictionary containing measurements and their corresponding points.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # Draw the face mesh using Mediapipe landmarks
    draw_face_mesh(image, landmarks_extractor.landmarks)

    # Draw the forehead point using dlib
    draw_forehead_point(image, landmarks_extractor.forehead)
    
    # Draw measurement lines between consecutive points in each measurement list
    for key, points in measurements.items():
        if key == "Face Contour":
            # Draw the contour line without labels
            for i in range(len(points) - 1):
                draw_measurement_line(image, points[i], points[i + 1])  # No label
            # Optionally close the contour if needed
            draw_measurement_line(image, points[-1], points[0])  # Connect last to first, no label

        elif len(points) == 2:
            # Draw only once for pairs, without labels
            draw_measurement_line(image, points[0], points[1])
        elif len(points) > 1:
            # For other lines with multiple points, connect consecutive points without labels
            for i in range(len(points) - 1):
                new_draw_measurement_line(image, points[i], points[i + 1])
    
    # Show the image with visualizations
    cv2.imshow("Face Measurements Visualization", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
