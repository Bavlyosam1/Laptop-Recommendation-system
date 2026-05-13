# Image Captioning using ViT-GPT2 and EfficientNetB1-GRU

## Overview

This project focuses on automatic image caption generation using Deep Learning and Transformer-based architectures.

We implemented and compared two different image captioning models:

1. **ViT-GPT2 Transformer Model**
2. **EfficientNetB1 + GRU Model**

The goal of the project is to generate meaningful textual descriptions for images from the Flickr30k dataset.

---

# Project Objectives

- Build an end-to-end Image Captioning system
- Explore Transformer-based architectures
- Compare CNN-RNN models with Vision Transformers
- Analyze model performance using BLEU Score and Loss Curves
- Perform Comparative Analysis between different architectures

---

# Dataset

## Flickr30k Dataset

The Flickr30k dataset contains:
- 30,000 images
- Multiple captions for each image

Dataset structure:
```bash
flickr30k_images/
results.csv
```

---

# Models Used

## 1) ViT-GPT2 Model

### Architecture

- Vision Transformer (ViT) for image feature extraction
- GPT2 for caption generation

### Advantages

- Better contextual understanding
- Strong language generation
- Transformer-based architecture

### Technologies

- HuggingFace Transformers
- PyTorch
- Vision Transformer
- GPT2

---

## 2) EfficientNetB1 + GRU Model

### Architecture

- EfficientNetB1 for image feature extraction
- GRU for sequence generation

### Advantages

- Lightweight architecture
- Faster training
- Lower computational cost

### Technologies

- TensorFlow / Keras
- EfficientNetB1
- GRU
- Transfer Learning

---

# Project Structure

```bash
project/
│
├── flickr30k_images/
├── vit_caption_results/
├── vit_gpt2_flickr30k/
│
├── vits_model.ipynb
├── vits_test.ipynb
├── EfficientNetB1_Model_Improved.ipynb
├── evaluation.ipynb
├── comparative_analysis.ipynb
│
├── tokenizer.pkl
├── max_length.pkl
├── eff_history.pkl
│
└── README.md
```

---

# Data Preprocessing

## Text Processing

- Tokenization
- Sequence Padding
- Vocabulary Creation
- Maximum Sequence Length Extraction

## Image Processing

- Image Resizing
- Normalization
- EfficientNet / ViT preprocessing

---

# Training Details

## ViT-GPT2

- Pretrained ViT Encoder
- Pretrained GPT2 Decoder
- Fine-Tuning on Flickr30k

## EfficientNetB1-GRU

- Transfer Learning using ImageNet weights
- Frozen layers + Fine-Tuning
- EarlyStopping
- ReduceLROnPlateau

---

# Evaluation Metrics

## BLEU Score

BLEU Score measures how similar generated captions are to reference captions.

Higher BLEU Score indicates better caption quality.

---

# Comparative Analysis

We compared both models using:

- Training Loss
- Validation Loss
- BLEU Score
- Generated Captions
- Training Stability
- Architecture Complexity

---

# Results

## ViT-GPT2

### Strengths
- Better sentence quality
- More natural captions
- Strong contextual understanding

### Weaknesses
- Slower training
- Higher computational cost

---

## EfficientNetB1-GRU

### Strengths
- Faster training
- Lightweight architecture
- Lower memory usage

### Weaknesses
- Simpler captions
- Less contextual understanding

---

# Example Output

## Input Image

Image of a dog running through grass.

### ViT-GPT2 Caption
> "A dog running through the grass"

### EfficientNetB1-GRU Caption
> "Dog running in grass"

---

# Loss Curves

The project includes:
- Training Loss Curves
- Validation Loss Curves
- Comparative Visualizations

---

# Technologies Used

## Languages
- Python

## Libraries
- TensorFlow
- Keras
- PyTorch
- Transformers
- NumPy
- Pandas
- Matplotlib
- PIL

---

# Key Deep Learning Concepts

This project demonstrates understanding of:

- CNNs
- Vision Transformers (ViT)
- GPT2
- GRU
- Transfer Learning
- Fine-Tuning
- Tokenization
- Sequence Modeling
- Attention Mechanisms
- Generative Models

---

# Installation

## Clone Repository

```bash
git clone <your-github-repo-link>
cd project
```

---

# Install Dependencies

```bash
pip install tensorflow
pip install transformers
pip install torch
pip install matplotlib
pip install pandas
pip install pillow
```

---

# Run Notebooks

```bash
jupyter notebook
```

Run:
- `vits_model.ipynb`
- `EfficientNetB1_Model_Improved.ipynb`
- `evaluation.ipynb`
- `comparative_analysis.ipynb`

---

# Future Improvements

- Use larger datasets
- Improve BLEU score
- Add Beam Search
- Add Attention Visualization
- Deploy as Web Application

---

# Conclusion

This project successfully compares Transformer-based and CNN-RNN based Image Captioning systems.

The Transformer-based ViT-GPT2 model achieved stronger language generation and contextual understanding, while EfficientNetB1-GRU provided faster and lighter training performance.

The comparative analysis demonstrates the strengths and trade-offs between modern Transformer architectures and traditional CNN-RNN pipelines.

---

# Authors

- Your Name
- Team Members Names

---

# Course

Deep Learning / Artificial Intelligence Course Project

---

# License

This project is for educational purposes only.
