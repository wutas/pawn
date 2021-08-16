import os
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

DATA_PATH = r"images"

NUM_WORKERS = 2

SIZE_H = SIZE_W = 96 * 2

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

transformer = transforms.Compose([
    transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
    transforms.ToTensor(),  # converting to tensors
    transforms.Normalize(image_mean, image_std)  # normalize image data per-channel
])

loader = transforms.Compose([
    transforms.ToTensor()])

st.title("Классификация украшений")

uploaded_file = st.file_uploader("Загрузите фотографию украшения...", type="jpg")
model_type = st.sidebar.radio('Select model type', ['Model_v1', 'Model_v2', 'Composite'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # image.save('images/real_photo/unk1.jpg')
    # image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = transformer(image).unsqueeze(0)

    st.write("")
    gif_runner = st.image('wait4.gif')

    model_bin = torch.load('model_resnet_152_finetune.ckpt',
                           map_location=torch.device(device))
    model_bin_v2 = torch.load('jew_or_not.ckpt',
                              map_location=torch.device(device))

    model_mult = torch.load('5_class_77_v2.ckpt',
                            map_location=torch.device(device))
    model_mult_v2 = torch.load('5_class_bigdata.ckpt',
                               map_location=torch.device(device))

    model_stone = torch.load('stone_no_stone.ckpt',
                             map_location=torch.device(device))

    model_silver_gold = torch.load('silver_gold.ckpt',
                                   map_location=torch.device(device))
    with torch.no_grad():
        X = image.to(device, torch.float)
        logits = model_bin.forward(X)
        logits_v2 = model_bin_v2.forward(X)
        scores = F.softmax(logits, 1).detach().cpu().numpy().tolist()
        scores_v2 = F.softmax(logits_v2, 1).detach().cpu().numpy().tolist()
        labels_hat_bin = np.argmax(scores, axis=1)
        labels_hat_bin_v2 = np.argmax(scores_v2, axis=1)

        if model_type == 'Model_v2':
            labels_hat_bin = labels_hat_bin_v2

            if max(scores_v2[0]) <= 0.6:
                st.write("I'm not sure, but ...")

        if model_type == 'Composite':
            scores_combine = np.array(scores) * 0.4 + np.array(scores_v2) * 0.6
            labels_hat_bin = np.argmax(scores_combine, axis=1)

            if max(scores_combine[0]) <= 0.6:
                st.write("I'm not sure, but ...")

        if model_type == 'Model_v1':
            if max(scores[0]) <= 0.6:
                st.write("I'm not sure, but ...")

        if labels_hat_bin == [0]:

            logits = model_mult.forward(X)
            logits_v2 = model_mult_v2.forward(X)
            scores = F.softmax(logits, 1).detach().cpu().numpy().tolist()
            scores_v2 = F.softmax(logits_v2, 1).detach().cpu().numpy().tolist()
            labels_hat_multi = np.argmax(scores, axis=1)
            labels_hat_multi_v2 = np.argmax(scores_v2, axis=1)

            logits_stone = model_stone.forward(X)
            scores_stone = F.softmax(logits_stone, 1).detach().cpu().numpy().tolist()
            labels_hat_stone = np.argmax(scores_stone, axis=1)

            logits_gold = model_silver_gold.forward(X)
            scores_gold = F.softmax(logits_gold, 1).detach().cpu().numpy().tolist()
            labels_hat_gold = np.argmax(scores_gold, axis=1)

            gif_runner.empty()
            # st.write("this is JEW")

            if model_type == 'Model_v2':
                labels_hat_multi = labels_hat_multi_v2

                if max(scores_v2[0]) <= 0.25:
                    st.write("I'm not sure, but ...")

            if model_type == 'Composite':
                scores_combine = np.array(scores) * 0.4 + np.array(scores_v2) * 0.6
                labels_hat_multi = np.argmax(scores_combine, axis=1)

                if max(scores_combine[0]) <= 0.25:
                    st.write("I'm not sure, but ...")

            if model_type == 'Model_v1':
                if max(scores[0]) <= 0.25:
                    st.write("I'm not sure, but ...")

            if labels_hat_multi == [0]:
                st.write("This is bracelet")

            elif labels_hat_multi == [1]:
                st.write("This is earrings")

            elif labels_hat_multi == [2]:
                st.write("This is necklace")

            elif labels_hat_multi == [3]:
                st.write("This is ring")

            else:
                st.write("This is watch")

            if labels_hat_stone == [0]:
                st.write("There isn't stone")
            else:
                st.write("There is stone")

            if labels_hat_gold == [0]:
                st.write("This is gold")
            else:
                st.write("This is silver")

        else:
            gif_runner.empty()
            st.write("This is not jewelry, load another photo")
