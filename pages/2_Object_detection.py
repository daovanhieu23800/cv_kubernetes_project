import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

@st.cache_resource
def load_model():
    model = ...
    return model

@st.cache_resource
def load_feature_extractor():
    feature_extractor = ...
    return feature_extractor

def model_inference(uploaded_file):
    with torch.no_grad():
        image = Image.open(uploaded_file).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = labels[predicted_class_idx]
    # Get the predicted class label
    #print(predicted_class)
    return predicted_class

model = load_model()
feature_extractor = load_feature_extractor()
labels = model.config.id2label

def main():
    st.markdown('# Welcome to Object Detection')
    st.markdown('In this page you can detect an object in video or image ')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        prediction = model_inference(uploaded_file)
        st.markdown(f"# This image is a `{prediction}`")
    return
    
if __name__ == "__main__":
    main()