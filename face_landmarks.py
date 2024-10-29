import cv2
import mediapipe as mp
import dlib
import os
from pathlib import Path

# Dictionary to store landmark indices for different facial regions
FACE_LANDMARKS = {
    # "forehead": [10],  # This will be handled using dlib with 81 landmarks
    "left_forehead_point": [54],
    "right_forehead_point": [284],
    "left_eye": [33, 133, 159],
    "right_eye": [263, 362, 386],
    "nose": [1, 2, 98, 327, 195],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405],
    "chin": [152],
    "left_cheek": [227],
    "right_cheek": [454],
    "left_jaw_points": [93, 132, 58, 138, 172, 136, 149, 176, 148],
    "right_jaw_points": [377, 400, 378, 379, 365, 364, 367, 397, 288, 433, 352],
    "left_eye_center": [468],
    "right_eye_center": [473],
    "chin_upperpoint": [17],
    "upper_lip": [13, 14],  # Upper lip center
    "nose_top": [168],  # Top of the nose
    "nose_bottom": [6, 197, 195],  # Base of the nose
    "left_cheekbone": [234],  # Example index for left cheekbone
    "right_cheekbone": [454],  # Example index for right cheekbone
    "left_eye_outer": [33, 133],  # Outer landmarks of the left eye
    "left_eye_inner": [159, 145],  # Inner landmarks of the left eye
    "right_eye_outer": [263, 362],  # Outer landmarks of the right eye
    "right_eye_inner": [386, 374],  # Inner landmarks of the right eye
    "upper_lip_outer": [61, 81, 13],  # Outer upper lip landmarks
    "upper_lip_inner": [0, 17, 314],  # Inner upper lip landmarks
    "lower_lip_outer": [84, 17, 405],  # Outer lower lip landmarks
    "lower_lip_inner": [91, 181, 84],  # Inner lower lip landmarks
    # Added 'face_contour' to include all landmarks required for rectangularity
    "face_contour": [10, 338, 297, 332, 284, 251, 389, 356, 323, 361, 
                    127, 162, 21, 103, 67, 109, 288, 397, 365, 379, 
                    378, 400, 377, 152, 148, 176, 149, 150, 136, 
                    172, 58, 132, 93, 234, 454],
    
    "outer_lip_indices": [61, 291],  # Left and right corners of the mouth
    "upper_outer_lip_index": [0],      # Upper outer lip edge
    "lower_outer_lip_index": [17],     # Lower outer lip edge
    "lip_left_corner": [61],
    "lip_right_corner": [291],
}

# Initialize dlib's face detector and shape predictor
DLIB_FACE_DETECTOR = dlib.get_frontal_face_detector()
DLIB_SHAPE_PREDICTOR = dlib.shape_predictor(
    "shape_predictor_81_face_landmarks.dat"
)  # Update path to 81 landmark model file

class FaceLandmarks:
    """
    A class to handle extraction and access of facial landmarks using Mediapipe and dlib.
    """

    def __init__(self, image_path):
        """
        Initialize the Landmarks class with the image path and extract landmarks.

        Parameters:
            image_path (str): Path to the image file containing the face.
        """
        self.image_path = image_path
        self.landmarks = self._extract_landmarks()
        self.forehead_point = self._extract_forehead_point_dlib()  # Extract forehead point using dlib

    def _extract_landmarks(self):
        """
        Extract facial landmarks from the image using Mediapipe.

        Returns:
            list: A list of tuples representing the coordinates of all detected landmarks.
                  Returns an empty list if no landmarks are detected.
        """
        # Load the image
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error: Unable to load image '{self.image_path}'. Please check the file path.")
            return []

        # Convert the image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize Mediapipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_image)

            # If landmarks are detected, extract them
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                height, width, _ = image.shape
                # Convert normalized coordinates to pixel coordinates
                landmarks = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks.landmark]
                return landmarks

        return []  # Return an empty list if no landmarks are detected

    def _extract_forehead_point_dlib(self):
        """
        Extract the forehead point using dlib with 81 landmarks model.

        Returns:
            tuple: Coordinates of the forehead point as (x, y). Returns None if not detected.
        """
        # Load the image
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"Error: Unable to load image '{self.image_path}' for dlib processing. Please check the file path.")
            return None

        # Convert to grayscale as dlib works with grayscale images
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = DLIB_FACE_DETECTOR(gray_image)

        # If faces are detected, get the landmarks for the first face
        if faces:
            face = faces[0]
            shape = DLIB_SHAPE_PREDICTOR(gray_image, face)

            # Assuming the forehead point is the 71st point in the 81 landmarks model
            try:
                forehead_point = (shape.part(71).x, shape.part(71).y)
                adjusted_forehead_point = (forehead_point[0], forehead_point[1] - 12)  # Adjust as needed
                return adjusted_forehead_point
            except IndexError:
                print(f"Error: Unable to access the forehead point in the landmarks for '{self.image_path}'.")
                return None

        return None  # Return None if no face is detected

    def _get_landmarks(self, region):
        """
        Internal method to get landmarks for a specific facial region.

        Parameters:
            region (str): The name of the facial region.

        Returns:
            list or tuple: A list of tuples representing the coordinates of the requested facial region,
                           or a tuple if a single point is returned.
        """
        if region == "forehead":
            # Return the dlib-extracted forehead point directly as a tuple
            return self.forehead_point if self.forehead_point else []
        
        indices = FACE_LANDMARKS.get(region, [])
        if not indices:
            print(f"Warning: No landmarks defined for region '{region}'.")
            return []
        
        if len(indices) == 1:
            try:
                return self.landmarks[indices[0]]
            except IndexError:
                print(f"Error: Landmark index {indices[0]} is out of bounds for image '{self.image_path}'.")
                return []
       
        try:
            return [self.landmarks[i] for i in indices]
        except IndexError as e:
            print(f"Error accessing landmark indices for region '{region}' in image '{self.image_path}': {e}")
            return []

    # Property methods for accessing different facial regions
    @property
    def forehead(self):
        return self._get_landmarks("forehead")
    
    @property
    def chin_upperpoint(self):
        return self._get_landmarks("chin_upperpoint")
    
    @property
    def left_eye_center(self):
        return self._get_landmarks("left_eye_center")
    
    @property
    def right_eye_center(self):
        return self._get_landmarks("right_eye_center")
    
    @property
    def left_forehead_point(self):
        return self._get_landmarks("left_forehead_point")
    
    @property
    def right_forehead_point(self):
        return self._get_landmarks("right_forehead_point")
    
    @property
    def right_jaw_point(self):
        return self._get_landmarks("right_jaw_points")

    @property
    def left_jaw_point(self):
        return self._get_landmarks("left_jaw_points")

    @property
    def left_eye(self):
        return self._get_landmarks("left_eye")

    @property
    def right_eye(self):
        return self._get_landmarks("right_eye")

    @property
    def nose(self):
        return self._get_landmarks("nose")

    @property
    def mouth(self):
        return self._get_landmarks("mouth")
    
    @property
    def left_cheek(self):
        return self._get_landmarks("left_cheek")
    
    @property
    def right_cheek(self):
        return self._get_landmarks("right_cheek")
    
    @property
    def chin(self):
        return self._get_landmarks("chin")
    
    @property
    def face_contour(self):
        return self._get_landmarks("face_contour")
    
    @property
    def upper_lip(self):
        return self._get_landmarks("upper_lip")

    @property
    def nose_top(self):
        return self._get_landmarks("nose_top")

    @property
    def nose_bottom(self):
        return self._get_landmarks("nose_bottom")

    @property
    def left_cheekbone(self):
        return self._get_landmarks("left_cheekbone")

    @property
    def right_cheekbone(self):
        return self._get_landmarks("right_cheekbone")

    @property
    def left_eye_outer(self):
        return self._get_landmarks("left_eye_outer")

    @property
    def left_eye_inner(self):
        return self._get_landmarks("left_eye_inner")

    @property
    def right_eye_outer(self):
        return self._get_landmarks("right_eye_outer")

    @property
    def right_eye_inner(self):
        return self._get_landmarks("right_eye_inner")

    @property
    def upper_lip_outer(self):
        return self._get_landmarks("upper_lip_outer")

    @property
    def upper_lip_inner(self):
        return self._get_landmarks("upper_lip_inner")

    @property
    def lower_lip_outer(self):
        return self._get_landmarks("lower_lip_outer")

    @property
    def lower_lip_inner(self):
        return self._get_landmarks("lower_lip_inner")
    
    @property
    def outer_lip_indices(self):
        return self._get_landmarks("outer_lip_indices")
    
    @property
    def upper_outer_lip_index(self):
        return self._get_landmarks("upper_outer_lip_index")
    
    @property
    def lower_outer_lip_index(self):
        return self._get_landmarks("lower_outer_lip_index")
    
    @property
    def lip_left_corner(self):
        return self._get_landmarks("lip_left_corner")
    
    @property
    def lip_right_corner(self):
        return self._get_landmarks("lip_right_corner")


    def get_all_landmarks(self):
        """
        Get all facial landmarks.

        Returns:
            list: A list of tuples representing the coordinates of all facial landmarks.
        """
        return self.landmarks

def draw_landmarks_on_image(image, landmarks, forehead_point=None, color=(0, 255, 0), radius=2, thickness=-1):
    """
    Draw facial landmarks on an image.

    Parameters:
        image (numpy.ndarray): The image on which to draw.
        landmarks (list): A list of (x, y) tuples representing landmark points.
        forehead_point (tuple, optional): The forehead point to draw.
        color (tuple): Color for the landmarks (B, G, R).
        radius (int): Radius of the landmark circles.
        thickness (int): Thickness of the circle outline. -1 fills the circle.

    Returns:
        numpy.ndarray: The image with landmarks drawn.
    """
    # Draw Mediapipe landmarks
    for point in landmarks:
        cv2.circle(image, point, radius, color, thickness)

    # Draw dlib forehead point if available
    if forehead_point:
        cv2.circle(image, forehead_point, radius+2, (0, 0, 255), thickness)  # Red color for forehead

    return image

def process_directory(input_dir, output_dir, color=(0, 255, 0), radius=2, thickness=-1):
    """
    Process all images in the input directory, mark facial landmarks, and save annotated images to the output directory.

    Parameters:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory where annotated images will be saved.
        color (tuple): Color for the landmarks (B, G, R).
        radius (int): Radius of the landmark circles.
        thickness (int): Thickness of the circle outline. -1 fills the circle.

    Returns:
        None
    """
    # Supported image extensions
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Iterate over all image files in the input directory
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() not in supported_extensions:
            print(f"Skipping unsupported file: {image_file.name}")
            continue

        print(f"Processing image: {image_file.name}")
        face_land_marks = FaceLandmarks(str(image_file))

        # Load the image using OpenCV
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"Error: Unable to load image '{image_file.name}'. Skipping.")
            continue

        # Get all landmarks
        landmarks = face_land_marks.get_all_landmarks()

        # Get the forehead point from dlib
        forehead_point = face_land_marks.forehead

        if not landmarks and not forehead_point:
            print(f"No landmarks detected for image '{image_file.name}'. Skipping.")
            continue

        # Draw landmarks on the image
        annotated_image = draw_landmarks_on_image(image, landmarks, forehead_point)

        # Optionally, draw connections between landmarks for better visualization
        # You can define connections as per your requirements
        # Example: Drawing face contour
        face_contour = face_land_marks.face_contour
        if face_contour:
            for i in range(len(face_contour)):
                start_point = face_contour[i]
                end_point = face_contour[(i + 1) % len(face_contour)]
                cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 1)  # Blue lines for contour

        # Define the output file path
        output_file = output_path / image_file.name

        # Save the annotated image
        cv2.imwrite(str(output_file), annotated_image)
        print(f"Saved annotated image to '{output_file}'")

if __name__ == "__main__":
    # Example usage
    input_directory = "combined_data/testing/Round"  # Update with your input directory
    output_directory = "landmark_images"  # Update with your output directory

    process_directory(input_directory, output_directory)
