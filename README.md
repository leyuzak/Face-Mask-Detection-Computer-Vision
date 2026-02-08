# Face Mask Detection – Computer Vision

🔗 **Kaggle Notebook (Training & Analysis)**  
https://www.kaggle.com/code/leyuzakoksoken/face-mask-detection-computer-vision  

🔗 **Live Demo (Hugging Face Space)**  
https://huggingface.co/spaces/leyuzak/Face-Mask-Detection-Computer-Vision  

---

## 📌 Project Overview

This project implements a **binary image classification system** that detects whether a person is **wearing a face mask or not** using deep learning and computer vision techniques.

- **Task:** Binary Classification (`with_mask` vs `no_mask`)
- **Domain:** Computer Vision
- **Model:** ResNet18 (PyTorch)
- **Deployment:** Hugging Face Spaces

---

## 🧠 Model & Training

- The model is based on **ResNet18**, pretrained on ImageNet and fine-tuned on a face mask dataset.
- Input images are resized to **224 × 224**.
- The final classification layer is adapted for two classes.
- The trained model is saved as a `.pth` file and reused for inference and deployment.

All training, evaluation, and experimentation were conducted on **Kaggle**.

---

## 🚀 Online Deployment

The trained model is deployed as an interactive web application.

Users can:
- Upload an image
- Instantly receive a prediction
- View class probabilities

The application is hosted on **Hugging Face Spaces**.

---

## 🖼️ Example Output

The model predicts:
- **with_mask**
- **no_mask**

along with confidence scores for each class.

