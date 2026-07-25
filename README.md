<div align="center">

# 🎭 DeepVision AI
### AI-Powered Deepfake Detection using Vision Transformers and Temporal Learning

<p align="center">
  <img src="https://i.ibb.co/7dtWBspY/logo.png">
</p>

<p align="center">
  <strong>Detecting AI-generated manipulated videos using Vision Transformers, Temporal Pooling, and Deep Learning.</strong>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![Torchvision](https://img.shields.io/badge/Torchvision-Latest-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-success)
![Research](https://img.shields.io/badge/Research-Final%20Year%20Project-blueviolet)

</p>

</div>

## 🚀 Overview

DeepVision AI is an advanced deepfake detection framework that leverages **Vision Transformers (ViT)** and **Temporal Feature Pooling** to distinguish authentic videos from AI-generated manipulations.

Unlike conventional CNN-based detectors that analyze individual frames independently, DeepVision AI learns both **spatial** and **temporal** inconsistencies across video sequences, making it more robust against modern deepfake generation techniques.

The project is designed for:

- 🎓 Academic Research
- 📄 IEEE Publications
- 💼 AI/ML Portfolio
- 🧠 Computer Vision Research
- 🔒 Media Authenticity Verification

---

# ✨ Features

- ✅ Vision Transformer (ViT-B/16)
- ✅ Temporal Feature Pooling
- ✅ Video Sequence Learning
- ✅ End-to-End Training Pipeline
- ✅ Automatic Checkpoint Saving
- ✅ CUDA GPU Support
- ✅ Mixed Dataset Training
- ✅ Evaluation Metrics
- ✅ Confusion Matrix
- ✅ ROC-AUC Analysis
- ✅ Modular Architecture
- ✅ Production Ready Codebase

---

# 🧠 Architecture

```
               Video Input
                     │
                     ▼
          Frame Extraction
                     │
                     ▼
            Face Detection
                     │
                     ▼
          Face Alignment
                     │
                     ▼
         Frame Sequence Builder
                     │
                     ▼
          Vision Transformer
              (ViT-B/16)
                     │
                     ▼
         Temporal Mean Pooling
                     │
                     ▼
        Fully Connected Layers
                     │
                     ▼
        Real  ←────────→  Fake
```

---

# 📂 Project Structure

```text
DeepVision-AI/
│
├── configs/
│
├── data/
│
├── outputs/
│
├── scripts/
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── preprocessing/
│   └── utils/
│
├── tests/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── environment.yml
└── README.md
```

---

# 📊 Dataset

The model is trained on multiple publicly available deepfake datasets.

| Dataset | Purpose |
|---------|----------|
| FaceForensics++ | Training |
| Celeb-DF v2 | Training & Testing |
| DFDC | Future Evaluation |

The datasets contain:

- Original Videos
- DeepFake Videos
- FaceSwap
- Neural Manipulations

---

# ⚙️ Tech Stack

## Programming

- Python

## Deep Learning

- PyTorch
- Torchvision

## Computer Vision

- OpenCV
- facenet-pytorch

## Model

- Vision Transformer (ViT)

## GPU

- NVIDIA CUDA

## Utilities

- NumPy
- Pandas
- tqdm
- YAML

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Arvindhbabu/DeepVision-AI.git

cd DeepVision-AI
```

Create environment

```bash
conda env create -f environment.yml

conda activate deepvision-ai
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
python train.py --config configs/default.yaml
```

---

# 📈 Evaluation

```bash
python evaluate.py \
--config configs/default.yaml \
--checkpoint outputs/runs/<run>/best_model.pth
```

---

# 📊 Metrics

The framework evaluates using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

# 🧩 Workflow

```
Raw Videos
      │
      ▼
Frame Extraction
      │
      ▼
Face Detection
      │
      ▼
Sequence Generation
      │
      ▼
Vision Transformer
      │
      ▼
Temporal Pooling
      │
      ▼
Classification
      │
      ▼
Prediction
```

---

# 🎯 Applications

- Fake News Detection
- Digital Forensics
- Social Media Verification
- Cybersecurity
- Law Enforcement
- Election Integrity
- Identity Protection
- Journalism

---

# 📌 Current Progress

- ✅ Data Pipeline
- ✅ Dataset Builder
- ✅ Sequence Loader
- ✅ Vision Transformer
- ✅ Temporal Pooling
- ✅ GPU Training
- ✅ Evaluation Pipeline
- 🚧 Performance Optimization
- 🚧 IEEE Publication

---

# 📚 Future Improvements

- Spatial Attention
- Temporal Attention
- XAI Explainability
- Grad-CAM Visualization
- Multi-Dataset Benchmarking
- Model Quantization
- Real-Time Webcam Detection
- REST API Deployment
- Mobile Inference
- ONNX Export

---

# 📄 Research Contributions

This project explores:

- Vision Transformers for Deepfake Detection
- Temporal Feature Aggregation
- Video Sequence Representation Learning
- Robust Binary Classification
- Transfer Learning on Large Vision Models

---

# 🤝 Contributing

Contributions are welcome!

Feel free to:

- Fork the repository
- Create feature branches
- Submit Pull Requests
- Report Issues

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Arvindh Babu V

Final Year B.Tech Artificial Intelligence & Data Science

Passionate about

- Artificial Intelligence
- Computer Vision
- Deep Learning
- Generative AI
- MLOps

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a Star!

**Made with ❤️ using Python, PyTorch and Vision Transformers**

</div>
