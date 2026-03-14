# Image Caption Generator

The **Image Caption Generator** is a deep learning project that automatically generates descriptive captions for images. It combines **Computer Vision** and **Natural Language Processing (NLP)** to understand the visual content of an image and generate human-like textual descriptions.

---

## 🧠 How It Works

The model follows a two-stage architecture:

### 1. Image Feature Extraction
A pretrained **Convolutional Neural Network (CNN)** such as **ResNet**, **VGG**, or **Inception** is used to extract visual features from input images.

In this project, the CNN encoder produces **feature maps of size (49, 2048)**.  
These feature maps represent **49 spatial regions of the image**, each encoded with **2048-dimensional deep features** capturing high-level semantic information.

### 2. Caption Generation
The extracted image features are then passed into a **sequence model (RNN/LSTM)** that generates captions **word by word**.

### Model Pipeline

Image → CNN Feature Extractor → **49 × 2048 Feature Map** → LSTM Caption Generator → **Generated Caption**

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook (Kaggle)

---

## 📊 Dataset

The model can be trained using widely used image captioning datasets:

- **Flickr8k**
- **Flickr30k**

These datasets contain images paired with multiple human-written captions used for training caption generation models.

---

## 📈 Future Improvements

- Upgrade feature extraction to **(196, 1024) feature maps** for finer spatial representation.
- Improve caption quality using **attention mechanisms**.
- Deploy the model as an **API** for real-world applications.

---

## 📒 Kaggle Notebook

You can view the full training process and implementation on Kaggle:

🔗 https://www.kaggle.com/code/ramashishpanchal/image-caption-generator
