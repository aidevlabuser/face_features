import math
from face_landmarks import FaceLandmarks  # Import the FaceLandmarks class
from utils.math_helper_functions import (
    euclidean_distance,
    angle_of_3points
)
import numpy as np
from utils.visualization import new_visualize_landmarks

def bounding_rectangle_area(points):
    """
    Calculate the area of the minimum bounding rectangle for given points.

    Parameters:
        points (list): A list of tuples representing the coordinates of the points.

    Returns:
        float: The area of the bounding rectangle in pixels squared.
    """
    if not points or len(points) < 2:
        return 0

    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    width = max_x - min_x
    height = max_y - min_y

    return height / width 

class FaceMeasurements:
    """
    A class to calculate various face measurements using the extracted landmarks
    based on the features from the paper.
    """

    def __init__(self, image_path):
        """
        Initialize with the image path and extract landmarks using FaceLandmarks class.
        
        Parameters:
            image_path (str): Path to the image file containing the face.
        """
        self.image_path = image_path
        self.landmarks_extractor = FaceLandmarks(self.image_path)
        self.landmarks = self.landmarks_extractor.get_all_landmarks()

    # Existing Properties

    @property
    def face_length(self):
        """
        Calculate the face length using the forehead point and the chin point.
        
        Returns:
            float: The face length in pixels.
        """
        forehead_point = self.landmarks_extractor.forehead  # Tuple (x, y)
        chin_point = self.landmarks_extractor.chin  # Tuple (x, y)
        face_length_pixels = euclidean_distance(forehead_point, chin_point)
        return face_length_pixels  # Measurements are in pixels

    @property
    def chin_height(self):
        """
        Calculate the chin height using chin and upper chin points.
        
        Returns:
            float: The chin height in pixels.
        """
        chin = self.landmarks_extractor.chin  # Tuple (x, y)
        chin_upperpoint = self.landmarks_extractor.chin_upperpoint  # Tuple (x, y)
        return euclidean_distance(chin, chin_upperpoint)

    @property
    def face_width(self):
        """
        Calculate the face width using left and right cheek landmarks.
        
        Returns:
            float: The face width in pixels.
        """
        left_cheek = self.landmarks_extractor.left_cheek  # Tuple (x, y)
        right_cheek = self.landmarks_extractor.right_cheek  # Tuple (x, y)
        return euclidean_distance(left_cheek, right_cheek)

    @property
    def forehead_width(self):
        """
        Calculate the forehead width using the left and right forehead points.
        
        Returns:
            float: The forehead width in pixels.
        """
        left_forehead_point = self.landmarks_extractor.left_forehead_point  # Tuple (x, y)
        right_forehead_point = self.landmarks_extractor.right_forehead_point  # Tuple (x, y)
        return euclidean_distance(left_forehead_point, right_forehead_point)

    @property
    def jaw_width(self):
        """
        Calculate the jaw width using jawline landmarks.
        
        Returns:
            float: The jaw width in pixels.
        """
        # Using specific jaw points for calculation
        left_jaw = self.landmarks_extractor.left_jaw_point[3]  # Example index
        right_jaw = self.landmarks_extractor.right_jaw_point[6]  # Example index

        # Calculate jaw width
        jaw_width_pixels = euclidean_distance(left_jaw, right_jaw)
        return jaw_width_pixels  # Measurements are in pixels

    @property
    def jaw_angle(self):
        """
        Calculate the jaw angle using the chin and jaw points.
        
        Returns:
            float: The jaw angle in degrees.
        """
        chin_point = self.landmarks_extractor.chin  # Tuple (x, y)
        left_jaw = self.landmarks_extractor.left_jaw_point
        right_jaw = self.landmarks_extractor.right_jaw_point

        # Calculate the angle using three points: left jaw, chin, right jaw
        jaw_angle = angle_of_3points(left_jaw[-1], chin_point, right_jaw[0])
        return jaw_angle

    @property
    def face_rectangularity(self):
        """
        Calculate the face rectangularity using all face contour points.

        Returns:
            float: The rectangularity of the entire face in pixels squared.
        """
        face_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 
            361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 
            176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 
            162, 21, 54, 103, 67, 109, 10
        ]
        face_contour_points = [self.landmarks[i] for i in face_contour_indices if i < len(self.landmarks)]
        return bounding_rectangle_area(face_contour_points)

    @property
    def middle_face_rectangularity(self):
        """
        Calculate the rectangularity of the middle face region.

        Returns:
            float: The rectangularity of the middle face in pixels squared.
        """
        middle_face_indices = [234, 454, 93, 132, 58, 172]
        middle_face_points = [self.landmarks[i] for i in middle_face_indices if i < len(self.landmarks)]
        return bounding_rectangle_area(middle_face_points)

    @property
    def forehead_rectangularity(self):
        """
        Calculate the rectangularity of the forehead region.

        Returns:
            float: The rectangularity of the forehead in pixels squared.
        """
        central_forehead_point = self.landmarks_extractor.forehead
        left_forehead_point = self.landmarks_extractor.left_forehead_point
        right_forehead_point = self.landmarks_extractor.right_forehead_point
        
        # Ensure all points are available
        if not central_forehead_point or not left_forehead_point or not right_forehead_point:
            return 0

        forehead_points = [central_forehead_point, left_forehead_point, right_forehead_point]
        return bounding_rectangle_area(forehead_points)

    @property
    def chin_angle(self):
        """
        Calculate the angle at the chin.

        Returns:
            float: The chin angle in degrees.
        """
        chin_point = self.landmarks_extractor.chin
        left_jaw = self.landmarks_extractor.left_jaw_point
        right_jaw = self.landmarks_extractor.right_jaw_point
        return angle_of_3points(left_jaw[0], chin_point, right_jaw[-1])

    @property
    def ratio_lower_to_middle_width(self):
        """
        Calculate the ratio of the lower face width to the middle face width.

        Returns:
            float: The ratio of lower face width over middle face width.
        """
        lower_face_width = euclidean_distance(
            self.landmarks_extractor.left_jaw_point[3],
            self.landmarks_extractor.right_jaw_point[6]
        )
        middle_face_width = euclidean_distance(
            self.landmarks_extractor.left_cheek,
            self.landmarks_extractor.right_cheek
        )
        return lower_face_width / middle_face_width if middle_face_width != 0 else 0

    @property
    def ratio_upper_to_middle_width(self):
        """
        Calculate the ratio of the upper face width to the middle face width.

        Returns:
            float: The ratio of upper face width over middle face width.
        """
        upper_face_width = euclidean_distance(
            self.landmarks_extractor.left_forehead_point,
            self.landmarks_extractor.right_forehead_point
        )
        middle_face_width = euclidean_distance(
            self.landmarks_extractor.left_cheek,
            self.landmarks_extractor.right_cheek
        )
        return upper_face_width / middle_face_width if middle_face_width != 0 else 0

    @property
    def difference_ratio_upper_lower(self):
        """
        Calculate the difference between the upper to middle width ratio and lower to middle width ratio.

        Returns:
            float: The difference between RTop and RBot.
        """
        return self.ratio_upper_to_middle_width - self.ratio_lower_to_middle_width

    @property
    def face_aspect_ratio(self):
        """
        Calculate the ratio of the width to the height of the face.

        Returns:
            float: The face aspect ratio.
        """
        face_width = euclidean_distance(
            self.landmarks_extractor.left_cheek,
            self.landmarks_extractor.right_cheek
        )
        face_height = euclidean_distance(
            self.landmarks_extractor.forehead,
            self.landmarks_extractor.chin
        )

        if face_width == 0:
            return 0
        return face_width / face_height

    # New Properties

    # 1. Chin to Lip Height
    @property
    def chin_to_lip_height(self):
        """
        Calculate the vertical distance from the chin to the upper lip.

        Returns:
            float: The chin to lip height in pixels.
        """
        chin_point = self.landmarks_extractor.chin
        upper_lip_point = self.landmarks_extractor.upper_lip
        if chin_point and upper_lip_point:
            return euclidean_distance(chin_point, upper_lip_point[0])
        return 0

    # 2. Top of Nose to Hairline Height
    @property
    def nose_to_hairline_height(self):
        """
        Calculate the vertical distance from the top of the nose to the hairline.

        Returns:
            float: The nose to hairline height in pixels.
        """
        nose_top = self.landmarks_extractor.nose_top
        forehead_point = self.landmarks_extractor.forehead
        if nose_top and forehead_point:
            return euclidean_distance(nose_top, forehead_point)
        return 0

    # 3. Jawline Angularity
    @property
    def jawline_angularity(self):
        """
        Calculate the angularity of the jawline based on multiple jaw points.

        Returns:
            float: The average jawline angularity in degrees.
        """
        left_jaw = self.landmarks_extractor.left_jaw_point
        right_jaw = self.landmarks_extractor.right_jaw_point
        chin_point = self.landmarks_extractor.chin

        if left_jaw and right_jaw and chin_point:
            angles = []
            # Calculate angles between consecutive jaw points on the left
            for i in range(1, len(left_jaw)):
                angle = angle_of_3points(left_jaw[i-1], left_jaw[i], chin_point)
                angles.append(angle)
            # Calculate angles between consecutive jaw points on the right
            for i in range(1, len(right_jaw)):
                angle = angle_of_3points(right_jaw[i-1], right_jaw[i], chin_point)
                angles.append(angle)
            if angles:
                return sum(angles) / len(angles)
        return 0

    # 4. Curve of the Chin
    @property
    def chin_curvature(self):
        """
        Calculate the curvature of the chin by comparing the chin landmarks to a straight line.

        Returns:
            float: The average deviation from the straight line in pixels.
        """
        chin_points = self.landmarks_extractor.left_jaw_point + self.landmarks_extractor.right_jaw_point
        if not chin_points:
            return 0
        # Define a straight line from leftmost to rightmost chin point
        left_chin = chin_points[0]
        right_chin = chin_points[-1]
        # Calculate deviation for each chin point
        deviations = []
        for point in chin_points[1:-1]:
            # Distance from point to the straight line
            distance = self._point_to_line_distance(point, left_chin, right_chin)
            deviations.append(distance)
        if deviations:
            return sum(deviations) / len(deviations)
        return 0

    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate the perpendicular distance from a point to a line.

        Parameters:
            point (tuple): The (x, y) coordinates of the point.
            line_start (tuple): The (x, y) coordinates of the start of the line.
            line_end (tuple): The (x, y) coordinates of the end of the line.

        Returns:
            float: The perpendicular distance in pixels.
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if denominator == 0:
            return 0
        return numerator / denominator

    # 5. Nose Size
    @property
    def nose_size(self):
        """
        Calculate the size of the nose based on the distance between nose bridge and base.

        Returns:
            float: The nose size in pixels.
        """
        nose_top = self.landmarks_extractor.nose_top
        nose_bottom_points = self.landmarks_extractor.nose_bottom
        if nose_top and nose_bottom_points:
            # Average distance from nose top to each bottom point
            distances = [euclidean_distance(nose_top, point) for point in nose_bottom_points]
            return np.mean(distances)
        return 0

    # 6. Eye Size and Spaces Between Them
    @property
    def left_eye_size(self):
        """
        Calculate the size of the left eye based on its landmarks.

        Returns:
            float: The left eye size in pixels.
        """
        left_eye_outer = self.landmarks_extractor.left_eye_outer
        left_eye_inner = self.landmarks_extractor.left_eye_inner
        if left_eye_outer and left_eye_inner:
            # Width and height of the eye
            eye_width = euclidean_distance(left_eye_outer[0], left_eye_outer[1])
            eye_height = euclidean_distance(left_eye_inner[0], left_eye_inner[1])
            return (eye_width + eye_height) / 2
        return 0

    @property
    def right_eye_size(self):
        """
        Calculate the size of the right eye based on its landmarks.

        Returns:
            float: The right eye size in pixels.
        """
        right_eye_outer = self.landmarks_extractor.right_eye_outer
        right_eye_inner = self.landmarks_extractor.right_eye_inner
        if right_eye_outer and right_eye_inner:
            eye_width = euclidean_distance(right_eye_outer[0], right_eye_outer[1])
            eye_height = euclidean_distance(right_eye_inner[0], right_eye_inner[1])
            return (eye_width + eye_height) / 2
        return 0

    @property
    def inter_eye_distance(self):
        """
        Calculate the distance between the centers of the two eyes.

        Returns:
            float: The inter-eye distance in pixels.
        """
        left_eye_center = self.landmarks_extractor.left_eye_center
        right_eye_center = self.landmarks_extractor.right_eye_center
        if left_eye_center and right_eye_center:
            return euclidean_distance(left_eye_center, right_eye_center)
        return 0

    # 7. Curve of Cheekbone (Angle)
    @property
    def cheekbone_angle(self):
        """
        Calculate the angle of the cheekbone based on cheekbone landmarks.

        Returns:
            float: The average cheekbone angle in degrees.
        """
        left_cheekbone = self.landmarks_extractor.left_cheekbone
        right_cheekbone = self.landmarks_extractor.right_cheekbone
        chin_point = self.landmarks_extractor.chin
        angles = []
        if left_cheekbone and chin_point and len(self.landmarks_extractor.left_jaw_point) > 0:
            angle = angle_of_3points(left_cheekbone, chin_point, self.landmarks_extractor.left_jaw_point[-1])
            angles.append(angle)
        if right_cheekbone and chin_point and len(self.landmarks_extractor.right_jaw_point) > 0:
            angle = angle_of_3points(right_cheekbone, chin_point, self.landmarks_extractor.right_jaw_point[-1])
            angles.append(angle)
        if angles:
            return sum(angles) / len(angles)
        return 0

    # 8. Size of Lips
    @property
    def lip_size(self):
        """
        Calculate the size of the lips based on mouth landmarks.

        Returns:
            float: The lip size in pixels.
        """
        # print(self.landmarks_extractor.upper_outer_lip_index)
        left_corner = self.landmarks_extractor.lip_left_corner
        right_corner = self.landmarks_extractor.lip_right_corner
        lip_width = math.hypot(right_corner[0] - left_corner[0], right_corner[1] - left_corner[1])

        upper_lip = self.landmarks_extractor.upper_outer_lip_index
        lower_lip = self.landmarks_extractor.lower_outer_lip_index
        lip_height = abs(lower_lip[1] - upper_lip[1])

        left_eye = self.landmarks_extractor.left_eye[0]
        right_eye = self.landmarks_extractor.right_eye[0]
        eye_distance = math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])

        normalized_lip_width = lip_width / eye_distance

        nose_tip = self.landmarks_extractor.nose[0]
        chin = self.landmarks_extractor.chin
        face_height = abs(chin[1] - nose_tip[1])

        normalized_lip_height = lip_height / face_height
        normalized_lip_size = normalized_lip_width * normalized_lip_height

        return normalized_lip_size

        # upper_lip_outer = self.landmarks_extractor.upper_lip_outer
        # upper_lip_inner = self.landmarks_extractor.upper_lip_inner
        # lower_lip_outer = self.landmarks_extractor.lower_lip_outer
        # lower_lip_inner = self.landmarks_extractor.lower_lip_inner
        # if upper_lip_outer and upper_lip_inner and lower_lip_outer and lower_lip_inner:
        #     # Calculate width and height of the lips
        #     upper_width = euclidean_distance(upper_lip_outer[0], upper_lip_outer[-1])
        #     lower_width = euclidean_distance(lower_lip_outer[0], lower_lip_outer[-1])
        #     upper_height = euclidean_distance(upper_lip_outer[1], upper_lip_inner[1])
        #     lower_height = euclidean_distance(lower_lip_outer[1], lower_lip_inner[1])
        #     return (upper_width + lower_width + upper_height + lower_height) / 4
        # return 0

    def get_all_measurements(self):
        """
        Get all measurements as defined in the paper and additional ones.

        Returns:
            dict: A dictionary containing all face measurements in pixels and degrees.
        """
        # Existing measurements
        measurements = {
            "face_rectangularity": self.face_rectangularity,
            "middle_face_rectangularity": self.middle_face_rectangularity,
            "forehead_rectangularity": self.forehead_rectangularity,
            "chin_angle": self.chin_angle,
            "ratio_lower_to_middle_width": self.ratio_lower_to_middle_width,
            "ratio_upper_to_middle_width": self.ratio_upper_to_middle_width,
            "difference_ratio_upper_lower": self.difference_ratio_upper_lower,
            "face_aspect_ratio": self.face_aspect_ratio,
            "face_ratio": round(self.face_length / self.face_width, 2) if self.face_width != 0 else 0,
            "forehead_ratio": round(self.forehead_width / self.face_width, 2) if self.face_width != 0 else 0,
            "cheekbone_ratio": round(self.face_width / self.face_width, 2),  # Always 1.00
            "jawline_ratio": round(self.jaw_width / self.face_width, 2) if self.face_width != 0 else 0,
            "chin_ratio": round(self.chin_height / self.face_length, 2) if self.face_length != 0 else 0,
            "jaw_angle": self.jaw_angle,
            "face_length": self.face_length,
            "face_width": self.face_width,
            "forehead_width": self.forehead_width,
            "jaw_width": self.jaw_width,
            "chin_height": self.chin_height
        }

        # New measurements
        measurements.update({
            "chin_to_lip_height": self.chin_to_lip_height,
            "nose_to_hairline_height": self.nose_to_hairline_height,
            "jawline_angularity": self.jawline_angularity,
            "chin_curvature": self.chin_curvature,
            "nose_size": self.nose_size,
            "left_eye_size": self.left_eye_size,
            "right_eye_size": self.right_eye_size,
            "inter_eye_distance": self.inter_eye_distance,
            "cheekbone_angle": self.cheekbone_angle,
            "lip_size": self.lip_size
        })

        # Print each measurement with appropriate formatting
        for key, value in measurements.items():
            if "angle" in key:
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f} degrees")
            elif "ratio" in key or "rectangularity" in key or "aspect_ratio" in key:
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")
            elif "distance" in key or "size" in key or "height" in key:
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f} pixels")
            elif "curvature" in key:
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f} pixels")
            else:
                print(f"{key.replace('_', ' ').capitalize()}: {value:.2f}")

        return measurements

# Example usage
if __name__ == "__main__":
    image_path = "train/Heart/Heart(0).jpg"  # Update the image path
    face_measurements = FaceMeasurements(image_path)
    all_measurements = face_measurements.get_all_measurements()
    # Optionally, visualize the landmarks and measurements
    new_visualize_landmarks(face_measurements.landmarks_extractor.landmarks, image_path,all_measurements)
