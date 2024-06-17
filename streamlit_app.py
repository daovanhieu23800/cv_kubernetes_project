import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor
import requests
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    return model

@st.cache_resource
def load_feature_extractor():
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    return feature_extractor

model = load_model()
feature_extractor = load_feature_extractor()

def get_file_content(file_path)-> str: 
    with open(file_path, 'r') as f:
        markdown_string = f.read()
    return markdown_string

def file_upload():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        prediction = model_inference(uploaded_file)
        st.markdown(f"# This image is a `{prediction}`")

def model_inference(uploaded_file):
    labels = requests.get('https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json').json()

    input_image = Image.open(uploaded_file).convert('RGB')
    input_image = feature_extractor(images=input_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input_image)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    predicted_class = labels[predicted_class_idx]
    return predicted_class

def run_the_app():
    file_upload()


    return 

def main():
    readme_text = st.markdown(get_file_content('./TEST.md'))
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Object detection"])
    
    if app_mode == "Show instructions":
        st.sidebar.success('You can choose different type of appication for to see what it can do.')
        
    elif app_mode == "Object detection":
        st.markdown('### This is where you play around with the model')
        readme_text.empty()
        run_the_app()

    return
    
if __name__ == "__main__":
    main()