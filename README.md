# Deep Learning Laboratory Works

This repository contains three comprehensive laboratory works covering fundamental deep learning concepts and implementations using TensorFlow/Keras. The labs progress from basic fully connected networks to advanced recurrent neural networks, providing hands-on experience with various neural network architectures and real-world applications.

## 📋 Overview

This project demonstrates practical implementations of various deep learning techniques across three major areas:

1. **Fully Connected Neural Networks** - Regression with hyperparameter optimization
2. **Convolutional Neural Networks** - Image classification with transfer learning
3. **Recurrent Neural Networks** - Text sentiment analysis with advanced RNN architectures

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow 2.x / Keras**
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Scikit-learn** - Data preprocessing and metrics
- **Google Colab** - Cloud-based training environment

## 📚 Laboratory Works

### [Lab 1: Fully Connected Neural Networks](./Lab1_README.md)

- **Dataset**: California Housing (Regression)
- **Focus**: Dense layers, optimization algorithms, regularization techniques
- **Key Concepts**: SGD, Adam optimization, dropout, batch normalization, hyperparameter tuning

### [Lab 2: Convolutional Neural Networks](./Lab2_README.md)

- **Dataset**: Custom image dataset (53 classes)
- **Focus**: CNN architectures, transfer learning, data augmentation
- **Key Concepts**: Conv2D layers, pooling, ImageDataGenerator, MobileNetV2, EfficientNet

### [Lab 3: Recurrent Neural Networks](./Lab3_README.md)

- **Dataset**: IMDb Movie Reviews (Sentiment Analysis)
- **Focus**: Text processing, RNN architectures, sequence modeling
- **Key Concepts**: Embeddings, LSTM, GRU, Bidirectional RNNs, tokenization

## 🎯 Learning Objectives

✅ **Neural Network Fundamentals**

- Understanding of neuron structure and network architecture
- Backpropagation and gradient descent algorithms
- Overfitting prevention techniques

✅ **Advanced Optimization**

- Adaptive methods (Adam, RMSprop, Adagrad)
- Learning rate scheduling
- Batch normalization and internal covariate shift

✅ **Computer Vision**

- Convolutional layer mechanics
- Popular CNN architectures (AlexNet, VGG, ResNet concepts)
- Transfer learning strategies
- Data augmentation techniques

✅ **Natural Language Processing**

- Text preprocessing and tokenization
- Word embeddings
- Sequence modeling with RNNs
- LSTM and GRU architectures
- Bidirectional processing

## 📊 Results Summary

| Lab   | Task Type            | Best Model                        | Accuracy/Performance   |
| ----- | -------------------- | --------------------------------- | ---------------------- |
| Lab 1 | Regression           | Dense Network with Regularization | MSE: ~0.35, MAE: ~0.45 |
| Lab 2 | Image Classification | EfficientNet Transfer Learning    | ~85-90% accuracy       |
| Lab 3 | Sentiment Analysis   | Bidirectional LSTM                | ~78-82% accuracy       |

## 🔧 Setup Instructions

1. **Environment Setup**:

   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```

2. **Google Colab** (Recommended):

   - Upload notebooks to Google Colab
   - Enable GPU/TPU acceleration
   - Run cells sequentially

3. **Local Setup**:
   - Ensure CUDA compatibility for GPU acceleration
   - Install required dependencies
   - Download datasets as specified in each lab

## 📁 File Structure

```
📦 Deep-Learning-Labs
├── 📜 README.md                           # Main project overview
├── 📓 Semyvolos_Lab1_DL_DigiJED.ipynb    # Lab 1: Fully Connected NNs
├── 📓 Lab2_Semyvolos_DL_DigiJED.ipynb    # Lab 2: Convolutional NNs
├── 📓 Lab3_Semyvolos_DL_DigiJED.ipynb    # Lab 3: Recurrent NNs
├── 📜 Lab1_README.md                      # Lab 1 detailed documentation
├── 📜 Lab2_README.md                      # Lab 2 detailed documentation
└── 📜 Lab3_README.md                      # Lab 3 detailed documentation
```

## 🎓 Key Achievements

- **Comprehensive Understanding**: Mastery of three fundamental deep learning paradigms
- **Practical Implementation**: Hands-on experience with real-world datasets
- **Performance Optimization**: Systematic hyperparameter tuning and architecture experimentation
- **Transfer Learning**: Successful application of pre-trained models for efficient training
- **End-to-End Pipeline**: Complete workflow from data preprocessing to model evaluation

## 🔍 Future Extensions

- Implementation of Transformer architectures for NLP tasks
- Advanced computer vision techniques (object detection, segmentation)
- Deployment of models using TensorFlow Serving or cloud platforms
- Experimentation with generative models (GANs, VAEs)

---

**Author**: Lev Semyvolos  
**Course**: Deep Learning  
**Year**: 2025

_This repository serves as a comprehensive demonstration of deep learning fundamentals and practical implementation skills developed throughout the course._
