# main.py

import os
import pandas as pd
from measurements.new_measurements import FaceMeasurements
from face_landmarks import FaceLandmarks



def facial_landmarks(image_path):
    
    face_landmarks = FaceLandmarks(image_path)
    return face_landmarks

def get_face_measurements(image_path):
    """
    Extract all face measurements from the given image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing all extracted measurements.
    """
    # Extract landmarks using the wrapper function
    face_landmarks = facial_landmarks(image_path)
    
    # Initialize FaceMeasurements with the extracted landmarks
    face_measurements = FaceMeasurements(image_path)
    
    # Get all measurements
    measurements = face_measurements.get_all_measurements()
    return measurements

def process_images_and_save_to_csv(image_directory, output_csv_path):
    """
    Process images in a directory structure, extract face measurements,
    and save the data to a CSV file.

    Parameters:
        image_directory (str): Directory containing subfolders with images.
        output_csv_path (str): Path to save the output CSV file.
    """
    data = []
    # Iterate through each subfolder in the image directory
    for folder_name in os.listdir(image_directory):
        folder_path = os.path.join(image_directory, folder_name)

        # Check if it's a directory
        if not os.path.isdir(folder_path):
            continue

        # True label is the folder name
        true_label = folder_name

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Check if it's an image file
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # Get the measurements
                measurements = get_face_measurements(image_path)

                # Append data to list
                data.append({
                    'Image Name': image_name,
                    'True Label': true_label,  # Optionally include the true label
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
                })

                print(f"Processed: {image_name}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data saved to {output_csv_path}")

if __name__ == "__main__":
    # Example usage for processing images and saving measurements to CSV

    # Directory path containing the labeled folders (update this path as needed)
    image_directory = "final_test"  # Update as per your directory structure

    # Output CSV file path
    csv_output_path = "final_test.csv"

    # Process images and save results to CSV
    process_images_and_save_to_csv(image_directory, csv_output_path)
    print(f"Results saved to {csv_output_path}")
