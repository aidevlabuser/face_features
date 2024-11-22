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
    features, image_with_landmark = get_face_measurements(image_path)
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

    return final_predictions, image_with_landmark


def predict_personality(features):
    # Rule-based mappings
    rules = [
        {'Face Shape': 'Oval', 'Face Wideness': 'Narrow', 'Face Contours': 'Soft Contours', 'Personality': 'Elegant'},
        {'Face Shape': 'Square', 'Jaw Lines': 'Sharp/Angular Jawline', 'Nose Wideness': 'Narrow Nose', 'Personality': 'Confident'},
        {'Face Shape': 'Round', 'Cheekbone Depth': 'Prominent Cheekbones', 'Lips Shape': 'Full Lips', 'Personality': 'Approachable'},
        {'Face Shape': 'Diamond', 'Jaw Lines': 'Tapered Jawline', 'Eyebrow Arch': 'High Arch', 'Personality': 'Charismatic'},
        {'Face Shape': 'Heart', 'Lips Shape': 'Bow-Shaped Lips', 'Eye Size': 'Large Eyes', 'Personality': 'Warm/Inviting'},
        {'Jaw Wideness': 'Wide Jaw', 'Face Contours': 'Sharp/Angular Contours', 'Cheekbone Depth': 'Flat Cheekbones', 'Personality': 'Bold'},
        {'Face Shape': 'Round', 'Eye Size': 'Average Eyes', 'Face Contours': 'Moderately Defined Contours', 'Personality': 'Friendly'},
        {'Face Shape': 'Oval', 'Cheekbone Depth': 'Sunken Cheekbones', 'Nose Type': 'Small Nose', 'Personality': 'Thoughtful'},
        {'Face Shape': 'Oblong', 'Lips Width': 'Wide Lips', 'Nose Shape': 'Button Nose', 'Personality': 'Cheerful'},
        {'Face Shape': 'Diamond', 'Jaw Lines': 'Rounded Jawline', 'Eyebrow Arch': 'Soft Arch', 'Personality': 'Approachable'},
        {'Face Shape': 'Square', 'Lips Shape': 'Thin Lips', 'Nose Shape': 'Hawk Nose', 'Personality': 'Powerful'},
        {'Face Shape': 'Oval', 'Face Wideness': 'Narrow', 'Eyebrow Arch': 'Straight (Flat) Brows', 'Personality': 'Subtle Elegance'},
        {'Face Shape': 'Round', 'Cheekbone Depth': 'Prominent Cheekbones', 'Jaw Wideness': 'Moderate Jaw', 'Personality': 'Compassionate'},
        {'Face Shape': 'Diamond', 'Lips Width': 'Average Lips', 'Jaw Lines': 'Tapered Jawline', 'Personality': 'Expressive'},
        {'Face Shape': 'Heart', 'Eye Size': 'Large Eyes', 'Eyebrow Arch': 'High Arch', 'Personality': 'Youthful'},
        {'Face Shape': 'Oblong', 'Face Contours': 'Tapered Contours', 'Lips Shape': 'Round Lips', 'Personality': 'Adaptable'},
        {'Jaw Wideness': 'Narrow Jaw', 'Lips Shape': 'Bow-Shaped Lips', 'Cheekbone Depth': 'Prominent Cheekbones', 'Personality': 'Creative'},
        {'Face Shape': 'Square', 'Face Wideness': 'Wide', 'Nose Shape': 'Flat Nose', 'Personality': 'Grounded'},
        {'Face Shape': 'Heart', 'Jaw Lines': 'Sharp/Angular Jawline', 'Lips Width': 'Wide Lips', 'Personality': 'Spirited'},
        {'Face Shape': 'Oval', 'Face Wideness': 'Extremely Wide', 'Eyebrow Arch': 'Moderate Arch', 'Personality': 'Balanced'},
        {'Face Shape': 'Oblong', 'Lips Shape': 'Wide Lips', 'Cheekbone Depth': 'Flat Cheekbones', 'Personality': 'Neutral'},
        {'Face Shape': 'Round', 'Nose Shape': 'Straight Nose', 'Jaw Wideness': 'Wide Jaw', 'Personality': 'Relatable'},
        {'Face Shape': 'Heart', 'Lips Shape': 'Flat Lips', 'Eye Size': 'Small Eyes', 'Personality': 'Subdued'},
        {'Face Shape': 'Square', 'Eyebrow Arch': 'Straight (Flat) Brows', 'Lips Width': 'Narrow Lips', 'Personality': 'Commanding'},
        {'Face Shape': 'Diamond', 'Face Contours': 'Moderately Defined Contours', 'Lips Shape': 'Full Lips', 'Personality': 'Radiant'},
        {'Face Shape': 'Oblong', 'Personality': 'Assertive'},
        {'Face Shape': 'Oval', 'Personality': 'Sophisticated'},
        {'Face Shape': 'Heart', 'Personality': 'Delicate'},
        {'Face Shape': 'Diamond', 'Personality': 'Reserved'},
        {'Face Shape': 'Square', 'Personality': 'Serious'},
    ]

    # Check each rule
    for rule in rules:
        # Match the rule if all specified features match
        if all(features.get(key) == value for key, value in rule.items() if key != 'Personality'):
            return rule['Personality']

    # Final fallback rule for unmatched features
    return 'Adaptive'


import pandas as pd
import streamlit as st
from PIL import Image

st.title("Image Processing App")
st.write("Upload an image to process and see the results!")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    filename = f"tmp/{uuid4()}.png"
    image.save(filename)

    result_dict, image_with_landmark = get_face_features(filename)
    image = Image.fromarray(image_with_landmark)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("### Personality:")
    personality = predict_personality(result_dict)
    st.write(personality)

    st.write("### Face features:")
    result_df = pd.DataFrame(list(result_dict.items()), columns=["Feature", "Prediction"])
    result_df.index += 1
    st.table(result_df)
