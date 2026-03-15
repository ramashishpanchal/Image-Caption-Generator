# Image Caption Generator

The **Image Caption Generator** is a deep learning project that automatically generates descriptive captions for images. It combines **Computer Vision** and **Natural Language Processing (NLP)** to understand the visual content of an image and generate human-like textual descriptions.

---

## 🧠 How It Works

The project follows a two-stage architecture consisting of an **image encoder** and a **caption generation model**.

### 1. Image Feature Extraction
A pretrained **Convolutional Neural Network (CNN)** such as **ResNet**, **VGG**, or **Inception** is used to extract visual features from input images.

In this project, the CNN encoder produces **feature maps of size (49, 2048)**.  
These feature maps represent **49 spatial regions of the image**, each encoded with **2048-dimensional deep feature vectors** that capture high-level semantic information.

### 2. Caption Generation

During the development of this project, the caption generation model evolved through two stages.

### 1. LSTM-based Caption Generator
The initial implementation used a **Recurrent Neural Network (LSTM)** as the decoder.  
In this approach, the model generated captions **word by word** using the image features extracted by the CNN encoder. The LSTM helped model sequential dependencies in the caption while conditioning on the visual features of the image.

### 2. Transformer-based Caption Generator
To further improve caption quality and contextual understanding, the architecture was later upgraded to a **Transformer-based decoder implemented using TensorFlow/Keras layers**.

Unlike LSTMs, the Transformer architecture uses **self-attention mechanisms** that allow the model to capture **long-range dependencies and richer contextual relationships** between words in the generated caption. This transition improved the model's ability to produce more coherent and contextually accurate descriptions.

---

### Model Pipeline

Image → CNN Feature Extractor → **49 × 2048 Feature Map** → Caption Generator (**LSTM / Transformer**) → **Generated Caption**

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
- Further improve caption generation using **attention optimization and larger transformer models**.
- Deploy the model as an **API** for real-world applications.

---

## 📒 Kaggle Notebook

You can view the full training process and implementation on Kaggle:

🔗 https://www.kaggle.com/code/ramashishpanchal/image-caption-generator
