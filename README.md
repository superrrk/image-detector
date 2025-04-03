# image-detector

# 🧠 Fake Image Detector

> A deep learning-powered web app that detects whether an image of a human face is AI-generated (fake) or real.

This project combines a custom PyTorch convolutional neural network (CNN) with a full-stack Flask + React frontend. Users can upload or select images, receive real-time predictions with confidence scores, and visualize saliency maps that explain the model's decision.

## 💻 Tech Stack

- **Frontend:** React, SCSS
- **Backend:** Flask, PyTorch, Captum (for saliency maps)
- **Model:** Custom CNN trained to classify real vs AI-generated faces (87% accuracy)
- **Other:** Numpy, Matplotlib, Pillow

## 📸 Screenshots + Live Demo

<img width="1501" alt="Image" src="https://github.com/user-attachments/assets/bb777c2d-05ca-4ba0-af15-1013fe77d1e6" />

---

## 🎯 Features

- Upload or drag-and-drop image input  
- Real-time classification of image as **REAL** or **FAKE**  
- Saliency map generation to visualize which image regions influenced the model  
- Confidence scores for each prediction  
- Sample image gallery (Ed Sheeran, Michelle Obama, etc.)  
- Light/dark mode toggle  
- Mobile responsive UI  

---

## 🧠 Model

A lightweight CNN was trained on real and GAN-generated face images using PyTorch. The model achieved **87% accuracy** on the validation set. Saliency maps were generated with **Captum** to explain the model's attention.

Training steps included:

- Dataset preprocessing and 80/20 train-validation split  
- Image normalization, resizing  
- 20-epoch training loop using CrossEntropyLoss + Adam optimizer  
- TorchVision for data loading and transforms
  
---

## 🚀 Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/fake-image-detector
   cd fake-image-detector
   ```

2. **Set up Python backend**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python server/app.py
   ```

3. **Set up React frontend**
   ```bash
   cd client
   npm install
   npm start
   ```

## 📚 Future Improvements

- Upload history with timestamps  
- Multiple model comparisons (e.g., VGG, ResNet)  
- User authentication + saved predictions  


