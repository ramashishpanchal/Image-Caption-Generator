Image Caption Generator

An Image Caption Generator is a deep learning project that automatically generates descriptive captions for images. It combines Computer Vision and Natural Language Processing (NLP) to understand image content and produce human-like descriptions.



🧠 How It Works

The project follows a two-part architecture:

1. Image Feature Extraction

A pretrained CNN (such as ResNet, VGG, or Inception) extracts visual features from the image.

In this project, the model is trained using feature maps of size (49, 2048) extracted from the CNN encoder.
These feature maps represent spatial regions of the image along with deep semantic information.

2. Caption Generation

An RNN/LSTM-based sequence model processes the extracted image features and generates captions word by word.

Pipeline:

Image → CNN Feature Extractor → 49×2048 Feature Map → Caption Model (LSTM) → Generated Caption



🛠️ Tech Stack

Python

TensorFlow 

NumPy

Matplotlib

Pandas

Keras

Jupyter Notebook (Kaggle)



📊 Dataset

Common datasets used for training:

Flickr8k

Flickr30k



📈 Future Improvements

Upgrade feature extraction to (196, 1024) feature maps for finer spatial representation

Deploy the model as an API


## 📒 Kaggle Notebook

You can view the full training and implementation in my Kaggle notebook:

🔗 https://www.kaggle.com/code/ramashishpanchal/image-caption-generator
