import os
from uuid import uuid4
import torch
import pickle
import torch.nn as nn
from face_measurments import get_face_measurements


os.makedirs('tmp', exist_ok=True)


target_columns = ["Face Shape", "Face Wideness", "Face Contours", "Jaw Wideness", "Jaw Lines", "Cheekbone Depth", "Nose Wideness", "Nose Shape", "Nose type", "Eye Size", "Lips Shape", "Lips Width", "Eyebrow Arch"]


class MultiOutputModel(nn.Module):
    def __init__(self):
        super(MultiOutputModel, self).__init__()
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(29, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # Output heads for each classification task
        self.face_shape = nn.Linear(32, 6)        # 6 classes for Face Shape
        self.face_wideness = nn.Linear(32, 4)     # 4 classes for Face Wideness
        self.face_contours = nn.Linear(32, 4)     # Adjust based on classes
        self.jaw_wideness = nn.Linear(32, 3)
        self.jaw_lines = nn.Linear(32, 5)
        self.cheekbone_depth = nn.Linear(32, 3)
        self.nose_wideness = nn.Linear(32, 2)
        self.nose_shape = nn.Linear(32, 6)
        self.nose_type = nn.Linear(32, 3)
        self.eye_size = nn.Linear(32, 3)
        self.lips_shape = nn.Linear(32, 8)
        self.lips_width = nn.Linear(32, 3)
        self.eyebrow_arch_shape = nn.Linear(32, 5)

    def forward(self, x):
        shared_output = self.shared(x)
        return {
            'Face Shape': self.face_shape(shared_output),
            'Face Wideness': self.face_wideness(shared_output),
            'Face Contours': self.face_contours(shared_output),
            'Jaw Wideness': self.jaw_wideness(shared_output),
            'Jaw Lines': self.jaw_lines(shared_output),
            'Cheekbone Depth': self.cheekbone_depth(shared_output),
            'Nose Wideness': self.nose_wideness(shared_output),
            'Nose Shape': self.nose_shape(shared_output),
            'Nose type': self.nose_type(shared_output),
            'Eye Size': self.eye_size(shared_output),
            'Lips Shape': self.lips_shape(shared_output),
            'Lips Width': self.lips_width(shared_output),
            'Eyebrow Arch': self.eyebrow_arch_shape(shared_output),
        }
    

model = MultiOutputModel()
model.load_state_dict(torch.load('face_features_trained_model.pth'))
model.eval()


def get_face_features(image_path):
    features = get_face_measurements(image_path)
    features = torch.tensor(list(features.values()), dtype=torch.float32).reshape(1, -1)

    label_encoders = {}
    for col in target_columns:
        with open(f'pkl/{col}_label_encoder.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)

    final_predictions = {}

    with torch.no_grad():
        predictions = model(features)

        for col, pred in predictions.items():
            pred_indices = torch.argmax(pred, dim=1)
            final_predictions[col] = label_encoders[col].inverse_transform(pred_indices.numpy())[0]

    return final_predictions




import pandas as pd
import streamlit as st
from PIL import Image

def process_image(image):
    # Replace this with your actual image processing logic
    # Here, we're just simulating processing with a sample dictionary
    result = {
        "Width": image.width,
        "Height": image.height,
        "Mode": image.mode,
        "Format": image.format,
        "Sample Feature": "Example Feature Value"
    }
    return result

st.title("Image Processing App")
st.write("Upload an image to process and see the results!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    filename = f"tmp/{uuid4()}.png"
    image.save(filename)
    
    result_dict = get_face_features(filename)

    st.write("### Result:")
    result_df = pd.DataFrame(list(result_dict.items()), columns=["Feature", "Prediction"])
    result_df.index += 1
    st.table(result_df)
