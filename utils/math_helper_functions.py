import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Function to calculate the slope between two points
def slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        return float('inf')  # Handle vertical line case
    return (y2 - y1) / (x2 - x1)

# Function to calculate the sum of slopes for a given set of points
def sum_slopes(points):
    total_slope = 0
    for i in range(len(points) - 1):
        total_slope += abs(slope(points[i], points[i + 1]))  # Summing absolute slopes
    return total_slope

# Function to calculate the sum of differences in Y-axis for a set of points
def sum_difference(points):
    total_difference = 0
    for i in range(len(points) - 1):
        total_difference += abs(points[i][1] - points[i + 1][1])
    return total_difference

# Function to calculate the angle between three points
def angle_of_3points(p1, p2, p3):
    # p2 is the middle point
    radian = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    degrees = math.degrees(abs(radian))
    return degrees

# math_helper_functions.py

def conversion_to_cm(pixel_distance, ref_pixel_distance):
    """
    Convert pixel distance to real-world centimeters using a fixed reference real-world distance.
    
    Parameters:
    pixel_distance (float): The distance measured in pixels.
    ref_pixel_distance (float): The reference distance in pixels (e.g., interpupillary distance in pixels).

    Returns:
    float: The real-world size in centimeters.
    """
    # Fixed reference real-world distance for interpupillary distance (average is 6.3 cm for adults)
    ref_real_distance_cm = 6.3
    
    if ref_pixel_distance == 0:
        raise ValueError("Reference pixel distance must be non-zero.")
    
    # Calculate the conversion factor
    conversion_factor = ref_real_distance_cm / ref_pixel_distance
    
    # Convert the pixel distance to centimeters
    real_distance_cm = pixel_distance * conversion_factor
    
    return real_distance_cm
