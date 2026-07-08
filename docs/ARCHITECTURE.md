# DeepVision AI v2.0 Architecture

> A Modular Explainable Deepfake Detection Framework using Vision Transformers, Temporal Modeling, and Explainable AI.

---

# Overview

DeepVision AI is a research-oriented and production-ready framework for detecting manipulated facial videos using deep learning.

The framework is designed around modular engineering principles, allowing datasets, models, training pipelines, explainability techniques, and deployment components to evolve independently.

---

# System Objectives

The primary objectives of DeepVision AI are:

- Detect AI-generated deepfake videos.
- Support multiple benchmark datasets.
- Provide explainable predictions using Grad-CAM.
- Enable reproducible deep learning experiments.
- Provide a deployable real-time inference system.

---

# System Architecture

```
                        DeepVision AI

                    Raw Video Dataset
                           │
                           ▼
                Dataset Index Generator
                           │
                           ▼
               outputs/dataset_index.csv
                           │
                           ▼
                 Sequence Dataset Loader
                           │
                           ▼
                 Balanced Data Sampler
                           │
                           ▼
                     PyTorch DataLoader
                           │
                           ▼
                     Training Engine
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 EfficientNet + BiLSTM          Vision Transformer
           │                               │
           └───────────────┬───────────────┘
                           ▼
                    Classification Head
                           │
                           ▼
                     Model Checkpoints
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
     Evaluation Engine                Inference Engine
          │                                 │
          ▼                                 ▼
      Performance                     Grad-CAM
      Metrics                         Visualization
          │                                 │
          └────────────────┬────────────────┘
                           ▼
                    Streamlit Application
```

---

# Repository Structure

```
DeepVision-AI/

├── app/
│
├── assets/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── processed/
│
├── docs/
│
├── outputs/
│
├── src/
│
├── tests/
│
├── train.py
├── evaluate.py
├── predict.py
└── README.md
```

---

# Source Code Architecture

```
src/

datasets/
    dataset_indexer.py
    sequence_dataset.py
    balanced_sampler.py
    dataloader.py

models/
    vit_temporal_pooling.py
    efficientnet_bilstm.py
    model_factory.py

training/
    trainer.py
    checkpoint.py
    early_stopping.py

evaluation/
    evaluator.py
    metrics.py

explainability/
    gradcam.py

inference/
    predictor.py

utils/
    config.py
    logger.py
```

---

# Data Pipeline

```
Raw Videos
      │
      ▼
Video Indexing
      │
      ▼
Sequence Generation (.npy)
      │
      ▼
Sequence Dataset
      │
      ▼
Balanced Sampling
      │
      ▼
PyTorch DataLoader
      │
      ▼
Model Training
```

---

# Model Pipeline

```
Sequence

↓

Vision Transformer

↓

Temporal Pooling

↓

Classification Head

↓

Prediction
```

---

# Training Pipeline

```
Load Configuration

↓

Load Dataset

↓

Create DataLoader

↓

Initialize Model

↓

Training Loop

↓

Validation

↓

Checkpoint Saving

↓

Performance Metrics
```

---

# Evaluation Pipeline

```
Model

↓

Test Dataset

↓

Predictions

↓

Accuracy

↓

Precision

↓

Recall

↓

F1 Score

↓

ROC Curve

↓

Confusion Matrix
```

---

# Explainability Pipeline

```
Input Sequence

↓

Forward Pass

↓

Target Layer

↓

Grad-CAM

↓

Heatmap

↓

Overlay

↓

Visualization
```

---

# Deployment Pipeline

```
Upload Video

↓

Face Extraction

↓

Sequence Generation

↓

Model Prediction

↓

Grad-CAM

↓

Confidence Score

↓

Download Report
```

---

# Engineering Principles

The project follows the following principles:

- Modular Architecture
- Configuration Driven
- Test Driven
- Reproducible Experiments
- Clean Separation of Responsibilities
- Research Friendly
- Deployment Ready

---

# Development Workflow

```
feature/*

↓

develop

↓

main
```

Each feature is implemented independently, tested, reviewed, and merged into the development branch before being promoted to the main branch.

---

# Supported Datasets

Current datasets:

- FaceForensics++
- Celeb-DF v2

Planned support:

- DFDC
- WildDeepfake
- ForgeryNet
- DeeperForensics-1.0

---

# Current Progress

## Sprint 1

- Repository Recovery
- Development Environment
- Configuration System

## Sprint 2

- Dataset Index Builder
- Sequence Dataset
- Dataset Metadata Generator

## Sprint 3

- Balanced Sampler (In Progress)
- DataLoader
- Training Engine

## Sprint 4

- Vision Transformer Integration
- EfficientNet-BiLSTM
- Model Factory

## Sprint 5

- Evaluation Pipeline

## Sprint 6

- Explainability

## Sprint 7

- Streamlit Deployment

---

# Future Improvements

- Multi-GPU Training
- Mixed Precision Training
- Distributed Training
- Self-Supervised Pretraining
- Transformer Ensemble
- Model Quantization
- ONNX Export
- TensorRT Deployment

---

# Citation

If you use this framework in research, please cite the repository after publication.

---

# License

This project will be released under the MIT License.