import torch
import torch.nn as nn
import cv2
import numpy as np
import gradio as gr
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from huggingface_hub import hf_hub_download

repo_id = "kullanici_adin/mask-classification-resnet18"
ckpt_path = hf_hub_download(
    repo_id=repo_id,
    filename="mask_cls_resnet18.pth"
)

ckpt = torch.load(ckpt_path, map_location="cpu")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

labels = ckpt["labels"]

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = tfms(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return labels[pred]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(),
    title="Face Mask Classification",
    description="Image-level binary classification: with_mask vs no_mask"
)

if __name__ == "__main__":
    demo.launch()
