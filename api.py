from typing import List, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# class JewImage(BaseModel):
#     image: Image.Image


def tensor_to_lable(model, X):
    logits = model.forward(X)
    scores = F.softmax(logits, 1).detach().cpu().numpy().tolist()
    labels_hat = np.argmax(scores, axis=1)
    return labels_hat


model_bin_v2 = torch.load('models/jew_or_not.ckpt',
                          map_location=torch.device(device))

model_mult_v2 = torch.load('models/5_class_bigdata.ckpt',
                           map_location=torch.device(device))

model_stone = torch.load('models/stone_no_stone.ckpt',
                         map_location=torch.device(device))

model_silver_gold = torch.load('models/silver_gold.ckpt',
                               map_location=torch.device(device))

app = FastAPI()

SIZE_H = SIZE_W = 96 * 2

image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]

transformer = transforms.Compose([
    transforms.Resize((SIZE_H, SIZE_W)),  # scaling images to fixed size
    transforms.ToTensor(),  # converting to tensors
    transforms.Normalize(image_mean, image_std)  # normalize image data per-channel
])

matching = {0: 'bracelet', 1: 'earrings', 2: 'necklace',
            3: 'ring', 4: 'watch'}


@app.post("/")
def search(uploaded_file: UploadFile = File(...)):
    image = Image.open(uploaded_file.file).convert('RGB')

    with torch.no_grad():
        image = transformer(image).unsqueeze(0)
        X = image.to(device, torch.float)
        labels_hat_bin_v2 = tensor_to_lable(model_bin_v2, X)

        if labels_hat_bin_v2 == [0]:
            labels_hat_multi_v2 = tensor_to_lable(model_mult_v2, X)[0]
            type_class = matching[labels_hat_multi_v2]

            labels_hat_stone = tensor_to_lable(model_stone, X)
            if labels_hat_stone == [0]:
                stone = 'NO'

            labels_hat_gold = tensor_to_lable(model_silver_gold, X)

            if labels_hat_gold == [0]:
                metal = 'gold'
            else:
                metal = 'silver'

            result = {'jewelry': 'YES', 'type': type_class, 'metal': metal, 'stone': stone}
        else:
            result = {'jewelry': 'NO', 'type': None, 'metal': None, 'stone': None}

    return {"results": result}
