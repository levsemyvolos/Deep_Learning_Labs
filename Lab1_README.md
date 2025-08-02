# Lab 1: Fully Connected Neural Networks

## 🎯 Objective

Explore the fundamentals of fully connected neural networks through practical implementation using TensorFlow/Keras. This lab focuses on regression tasks, hyperparameter optimization, and various regularization techniques to prevent overfitting.

## 📊 Dataset

**California Housing Dataset**

- **Size**: 20,640 samples
- **Features**: 8 numerical features
  1. MedInc (Median Income)
  2. HouseAge (House Age)
  3. AveRooms (Average Rooms)
  4. AveBedrms (Average Bedrooms)
  5. Population (Population)
  6. AveOccup (Average Occupancy)
  7. Latitude (Latitude)
  8. Longitude (Longitude)
- **Target**: Median house value (regression task)
- **Task Type**: Regression
- **Evaluation Metrics**: MSE (Mean Squared Error), MAE (Mean Absolute Error)

## 🏗️ Architecture Experiments

### 1. Baseline Models

- **Simple Model**:
  - Input → Dense(16, ReLU) → Dense(1)
  - Minimal architecture for baseline performance

### 2. Deeper Networks

- **Multi-layer Model**:
  - Input → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1)
  - Exploration of depth impact on performance

### 3. Activation Function Comparison

- **ReLU vs Tanh**:
  - Systematic comparison of activation functions
  - Analysis of convergence speed and final performance

### 4. Regularization Techniques

#### L2 Regularization

```python
Dense(64, activation='relu', kernel_regularizer=l2(0.001))
```

- Weight decay to prevent overfitting
- Multiple λ values tested (0.001, 0.01)

#### L1+L2 Regularization

```python
Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))
```

- Combined L1 and L2 penalties
- Feature selection + weight decay

#### Dropout

```python
Dropout(0.3)  # 30% dropout rate
```

- Random neuron deactivation during training
- Multiple dropout rates evaluated

#### Batch Normalization

```python
BatchNormalization()
```

- Internal covariate shift reduction
- Improved training stability

## 🔧 Optimization Algorithms

### 1. SGD (Stochastic Gradient Descent)

- Basic optimization with momentum
- Learning rate scheduling experiments

### 2. Adam Optimizer

```python
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
```

- Adaptive learning rates
- Superior performance in most experiments

### 3. RMSprop

- Adaptive learning rate method
- Good performance on regression tasks

## 📈 Training Configuration

### Data Preprocessing

- **Feature Scaling**: StandardScaler normalization
- **Train/Test Split**: 80%/20% ratio
- **Validation**: 20% of training data for validation

### Training Parameters

- **Batch Size**: 32, 64, 128 (experimented)
- **Epochs**: 100-200 with early stopping
- **Early Stopping**: Patience of 20 epochs
- **Loss Function**: Mean Squared Error
- **Metrics**: MSE, MAE

### Learning Rate Strategies

- **Fixed Learning Rate**: 0.001, 0.01
- **Learning Rate Decay**: Exponential and step decay
- **Adaptive Methods**: Adam's built-in adaptation

## 📊 Key Results

| Model Configuration   | Test MSE | Test MAE | Training Time |
| --------------------- | -------- | -------- | ------------- |
| Baseline (Simple)     | 0.52     | 0.53     | 2 min         |
| Deep Network          | 0.38     | 0.47     | 5 min         |
| + L2 Regularization   | 0.35     | 0.45     | 5 min         |
| + Dropout             | 0.36     | 0.46     | 6 min         |
| + Batch Normalization | 0.34     | 0.44     | 7 min         |
| Optimized (Best)      | **0.33** | **0.43** | 8 min         |

## 🔍 Key Findings

### 1. Architecture Impact

- **Depth vs Width**: Moderate depth (3-4 hidden layers) optimal
- **Layer Size**: Gradual decrease (64→32→16) worked well
- **Overfitting**: Deeper networks prone to overfitting without regularization

### 2. Regularization Effectiveness

- **L2 Regularization**: Most effective for preventing overfitting
- **Dropout**: Good generalization improvement
- **Batch Normalization**: Faster convergence and training stability
- **Combined Techniques**: Best results with multiple regularization methods

### 3. Optimization Insights

- **Adam > RMSprop > SGD**: Adam optimizer consistently outperformed others
- **Learning Rate**: 0.001 optimal for most configurations
- **Early Stopping**: Essential for preventing overfitting

### 4. Activation Functions

- **ReLU**: Faster training, better gradient flow
- **Tanh**: Slower convergence but sometimes better final performance
- **Dead Neurons**: ReLU occasionally suffered from dead neuron problem

## 🛠️ Implementation Highlights

### Model Building Function

```python
def create_model(layers, activation='relu', regularizer=None, dropout_rate=0.0):
    model = Sequential()
    for i, units in enumerate(layers):
        if i == 0:
            model.add(Dense(units, activation=activation,
                          input_shape=(n_features,),
                          kernel_regularizer=regularizer))
        else:
            model.add(Dense(units, activation=activation,
                          kernel_regularizer=regularizer))

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1))  # Output layer
    return model
```

### Training Pipeline

```python
def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    # Callbacks
    early_stopping = EarlyStopping(patience=20, restore_best_weights=True)

    # Training
    history = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=200,
                       batch_size=32,
                       callbacks=[early_stopping],
                       verbose=0)

    return model, history
```

## 📚 Theoretical Concepts Demonstrated

### 1. Backpropagation

- Gradient computation through chain rule
- Weight update mechanisms
- Vanishing/exploding gradient issues

### 2. Regularization Theory

- **Bias-Variance Tradeoff**: Regularization reduces variance
- **Occam's Razor**: Simpler models often generalize better
- **Capacity Control**: Limiting model complexity

### 3. Optimization Theory

- **Gradient Descent Variants**: SGD, momentum, adaptive methods
- **Learning Rate Impact**: Too high (divergence) vs too low (slow convergence)
- **Local Minima**: Non-convex optimization challenges

## 🎓 Learning Outcomes

✅ **Neural Network Fundamentals**

- Understanding of dense layer operations
- Forward and backward propagation implementation
- Loss function optimization

✅ **Hyperparameter Tuning**

- Systematic experimentation approach
- Performance evaluation metrics
- Model selection criteria

✅ **Regularization Mastery**

- Multiple regularization technique implementation
- Overfitting detection and prevention
- Generalization improvement strategies

✅ **Optimization Algorithms**

- Comparative analysis of optimizers
- Learning rate scheduling
- Convergence analysis

## 🔧 Usage Instructions

1. **Data Loading**:

   ```python
   from sklearn.datasets import fetch_california_housing
   housing = fetch_california_housing()
   ```

2. **Preprocessing**:

   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Model Training**:
   ```python
   model = create_optimized_model()
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   model.fit(X_train, y_train, validation_split=0.2)
   ```

## 🔮 Extensions and Improvements

- **Advanced Regularization**: Implement weight decay scheduling
- **Architecture Search**: Automated hyperparameter optimization
- **Ensemble Methods**: Combine multiple models for better performance
- **Feature Engineering**: Create polynomial features or interactions
- **Cross-Validation**: K-fold validation for robust evaluation

---

**Key Achievement**: Successfully demonstrated the impact of various neural network design choices on regression performance, achieving significant improvement through systematic experimentation and optimization techniques.
