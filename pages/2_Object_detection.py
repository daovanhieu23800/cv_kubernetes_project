import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


@st.cache_resource
def load_model():
    model = YOLO("yolov9s.pt")
    return model


def model_inference(image):
    with torch.no_grad():
        outputs = model(image)

    return outputs


def draw_image_with_boxes(image, results, obj_class="person", confident=0.5):

    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes

    for result in results:
        boxes = result.cpu().boxes.xyxy.numpy()  # Bounding boxes
        confs = result.cpu().boxes.conf.numpy()  # Confidence scores
        clss = result.cpu().boxes.cls.numpy()    # Class labels
        # print(clss)
        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = f'{result.names[int(cls)]}'
            color = (255, 0, 0)  # Green color for bounding box
            if (label == obj_class or obj_class == 'all') and (conf >= confident):
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    st.image(image, caption='Image with bounding box', use_column_width=True)


# Initialize a YOLO-World model
model = load_model()
object_names = list(model.names.values())
object_names.extend(['all'])
object_names = tuple(object_names)


def main():
    st.markdown('# Welcome to Object Detection')
    st.markdown('In this page you can detect an object in image ')
    uploaded_file = st.file_uploader("Choose a file")
    obj_class = st.sidebar.selectbox(
        "Choose object you want to detect",
        object_names
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        prediction = model_inference(image)
        draw_image_with_boxes(img_array, prediction,
                              obj_class, confidence_threshold)


if __name__ == "__main__":
    main()
