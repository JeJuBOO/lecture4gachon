import streamlit as st

import json
from io import BytesIO

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import timm



with open("imagenet_class_index.json", "r") as f:
    class_idx = json.load(f)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

available_models = [
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19_bn", "vgg19",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
]

def load_moel(model_name):
    model = timm.create_model(model_name, pretrained=True)
    return model

option = st.selectbox(
    'Select Model',
     available_models)
model = load_moel(option)
model.eval()

# load data
uploaded_file = st.file_uploader("Choose a Image")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(BytesIO(bytes_data)).convert("RGB")
    img_for_plot = np.array(image)
    
    img = transforms.ToTensor()(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    img = normalize(img).unsqueeze(dim=0)   
    result = model(img).squeeze(dim=0)
    predict_idx = result.argmax().item()
    prob = torch.softmax(result, dim=0)
    st.image(img_for_plot, use_column_width=True)
    st.text(f"{idx2label[predict_idx]}, {prob[predict_idx]}")