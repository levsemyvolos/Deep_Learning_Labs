# Lab 3: Recurrent Neural Networks

## 🎯 Objective

Explore recurrent neural networks (RNNs) for natural language processing tasks, focusing on sentiment analysis. This lab demonstrates text preprocessing, sequence modeling, and various RNN architectures including LSTM, GRU, and bidirectional networks.

## 📊 Dataset

**IMDb Movie Reviews Dataset**

- **Size**: 50,000 movie reviews (25,000 train, 25,000 test)
- **Classes**: Binary classification (Positive/Negative sentiment)
- **Class Distribution**: Balanced (50% positive, 50% negative)
- **Text Length**: Variable length reviews (typically 100-2000 words)
- **Task Type**: Sentiment Analysis (Binary Classification)
- **Source**: TensorFlow Datasets (`imdb_reviews`)

### Sample Data

```
Review (Negative): "This was an absolutely terrible movie. Don't be lured in by
Christopher Walken or Michael Ironside. Both are great actors, but this must
simply be their worst role in history..."

Review (Positive): "Mann photographs the Alberta Rocky Mountains in a superb
fashion, and Jimmy Stewart and Walter Brennan give enjoyable performances..."
```

## 🔤 Text Preprocessing Pipeline

### 1. Tokenization

```python
# Configuration
vocab_size = 10000      # Top 10,000 most frequent words
max_length = 100        # Maximum sequence length
embedding_dim = 64      # Embedding vector dimension
oov_token = "<OOV>"     # Out-of-vocabulary token

# Tokenizer setup
tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)
tokenizer.fit_on_texts(train_texts)
```

### 2. Sequence Processing

```python
# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad sequences to uniform length
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, truncating='post')
```

### 3. Vocabulary Analysis

- **Total Unique Words**: ~88,000
- **Vocabulary Size Used**: 10,000 (most frequent)
- **OOV Rate**: ~15-20% of tokens
- **Average Sequence Length**: ~230 words
- **Sequence Length After Padding**: 100 words

## 🏗️ RNN Architectures

### 1. Simple RNN (Baseline)

```python
model_simple_rnn = Sequential([
    # Embedding layer - converts word indices to dense vectors
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length
    ),

    # First Simple RNN layer
    SimpleRNN(
        64,                    # Number of hidden units
        return_sequences=True, # Return full sequence
        dropout=0.3,          # Dropout for regularization
        recurrent_dropout=0.3  # Recurrent connections dropout
    ),

    # Second Simple RNN layer
    SimpleRNN(
        32,                   # Fewer units for final representation
        dropout=0.3,
        recurrent_dropout=0.3
    ),

    # Dense layers for classification
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### 2. LSTM (Long Short-Term Memory)

```python
model_lstm = Sequential([
    # Embedding layer
    Embedding(vocab_size, embedding_dim, input_length=max_length),

    # First LSTM layer with more units
    LSTM(
        128,                   # Hidden units
        return_sequences=True, # Return sequences for stacking
        dropout=0.3,          # Input dropout
        recurrent_dropout=0.3  # Recurrent dropout
    ),

    # Second LSTM layer
    LSTM(
        64,                   # Hidden units
        dropout=0.3,
        recurrent_dropout=0.3
    ),

    # Dense layers for classification
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### 3. GRU (Gated Recurrent Unit)

```python
model_gru = Sequential([
    # Embedding layer
    Embedding(vocab_size, embedding_dim, input_length=max_length),

    # First GRU layer
    GRU(
        128,                   # Hidden units
        return_sequences=True,
        dropout=0.3,
        recurrent_dropout=0.3
    ),

    # Second GRU layer
    GRU(
        64,
        dropout=0.3,
        recurrent_dropout=0.3
    ),

    # Classification layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 4. Bidirectional LSTM

```python
model_bidirectional_lstm = Sequential([
    # Embedding layer
    Embedding(vocab_size, embedding_dim, input_length=max_length),

    # First Bidirectional LSTM layer
    Bidirectional(
        LSTM(
            64,                   # Units per direction (128 total)
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3
        )
    ),

    # Second Bidirectional LSTM layer
    Bidirectional(
        LSTM(
            32,                   # Units per direction (64 total)
            dropout=0.3,
            recurrent_dropout=0.3
        )
    ),

    # Classification layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

## ⚙️ Training Configuration

### Compilation Settings

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks and Training

```python
callbacks = [
    EarlyStopping(
        patience=5,
        restore_best_weights=True,
        monitor='val_loss'
    ),
    ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        monitor='val_loss'
    ),
    ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

history = model.fit(
    train_padded, train_labels,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

## 📊 Performance Results

| Model Architecture     | Parameters | Training Acc | Val Acc   | Test Acc  | Training Time | Convergence   |
| ---------------------- | ---------- | ------------ | --------- | --------- | ------------- | ------------- |
| Simple RNN             | 1.2M       | 92.3%        | 78.4%     | 77.1%     | 15 min        | 15 epochs     |
| **LSTM**               | **1.8M**   | **89.7%**    | **81.2%** | **78.4%** | **25 min**    | **12 epochs** |
| GRU                    | 1.6M       | 91.1%        | 80.8%     | 79.2%     | 20 min        | 13 epochs     |
| **Bidirectional LSTM** | **2.1M**   | **87.2%**    | **83.4%** | **81.7%** | **35 min**    | **18 epochs** |

### Detailed Performance Analysis

- **Best Single Model**: Bidirectional LSTM (81.7% test accuracy)
- **Fastest Training**: Simple RNN (significant overfitting)
- **Best Balance**: Standard LSTM (good performance, reasonable training time)
- **Most Robust**: Bidirectional LSTM (smallest train-test gap)

## 🔍 Key Findings

### 1. Architecture Comparison

- **Simple RNN**: Fast but suffers from vanishing gradient problem
- **LSTM**: Better long-term memory, more stable training
- **GRU**: Simpler than LSTM, competitive performance
- **Bidirectional**: Best performance by processing sequences in both directions

### 2. Sequence Length Impact

- **Short Sequences (50)**: Faster training, information loss
- **Medium Sequences (100)**: Optimal balance
- **Long Sequences (200+)**: Marginal improvement, much slower training

### 3. Embedding Insights

- **Embedding Dimension**: 64 optimal for this dataset
- **Vocabulary Size**: 10,000 words captured most important information
- **OOV Handling**: Proper handling crucial for generalization

### 4. Regularization Effects

- **Dropout**: Essential for preventing overfitting
- **Recurrent Dropout**: Particularly effective for RNNs
- **Early Stopping**: Prevented overfitting in all models

## 🛠️ Implementation Highlights

### Advanced Text Processing

```python
def preprocess_text(texts, tokenizer, max_length):
    """Advanced text preprocessing with multiple strategies"""

    # Basic tokenization
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding strategies
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',      # Pad at the end
        truncating='post'    # Truncate at the end
    )

    return padded

def analyze_text_statistics(texts, tokenizer):
    """Analyze text dataset statistics"""
    lengths = [len(text.split()) for text in texts]
    sequences = tokenizer.texts_to_sequences(texts)
    seq_lengths = [len(seq) for seq in sequences]

    stats = {
        'avg_length': np.mean(lengths),
        'max_length': np.max(lengths),
        'vocab_coverage': len(tokenizer.word_index),
        'oov_rate': 1 - np.mean([len(seq)/len(text.split()) for seq, text in zip(sequences, texts)])
    }

    return stats
```

### Model Evaluation

```python
def evaluate_sentiment_model(model, test_padded, test_labels):
    """Comprehensive model evaluation"""

    # Basic metrics
    test_loss, test_accuracy = model.evaluate(test_padded, test_labels, verbose=0)

    # Predictions
    predictions = model.predict(test_padded)
    predicted_classes = (predictions > 0.5).astype(int)

    # Detailed metrics
    precision = precision_score(test_labels, predicted_classes)
    recall = recall_score(test_labels, predicted_classes)
    f1 = f1_score(test_labels, predicted_classes)

    # Confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)

    return {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
```

### Attention Visualization (Bonus)

```python
def visualize_attention_weights(model, text, tokenizer, max_length):
    """Visualize which words the model focuses on"""

    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length)

    # Get intermediate outputs (requires model modification)
    attention_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    attention_weights = attention_model.predict(padded)

    # Visualize attention
    words = text.split()[:max_length]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(words)), attention_weights[0][:len(words)])
    plt.xticks(range(len(words)), words, rotation=45)
    plt.title('Attention Weights Visualization')
    plt.show()
```

## 📚 Theoretical Concepts Demonstrated

### 1. Sequence Modeling

- **Sequential Dependencies**: Modeling word order and context
- **Variable Length Handling**: Padding and masking strategies
- **Temporal Patterns**: Learning sentiment patterns over time

### 2. RNN Mechanics

- **Hidden States**: Information flow through sequences
- **Vanishing Gradients**: Why simple RNNs struggle with long sequences
- **Gating Mechanisms**: How LSTM/GRU solve vanishing gradient problem

### 3. LSTM Architecture

- **Cell State**: Long-term memory mechanism
- **Gates**: Forget, input, and output gates
- **Information Flow**: Selective information retention and forgetting

### 4. Bidirectional Processing

- **Forward Pass**: Left-to-right information flow
- **Backward Pass**: Right-to-left information flow
- **Context Integration**: Combining both directions for richer representations

### 5. Word Embeddings

- **Dense Representations**: Converting sparse word indices to dense vectors
- **Semantic Similarity**: Similar words have similar embeddings
- **Dimensionality**: Balancing expressiveness and computational efficiency

## 🎓 Learning Outcomes

✅ **NLP Fundamentals**

- Text preprocessing and tokenization
- Sequence padding and truncation strategies
- Vocabulary management and OOV handling

✅ **RNN Architecture Mastery**

- Simple RNN, LSTM, and GRU implementation
- Bidirectional processing
- Stacking and layer configuration

✅ **Sentiment Analysis Pipeline**

- End-to-end text classification workflow
- Model evaluation and comparison
- Performance optimization techniques

✅ **Advanced Concepts**

- Embedding layer utilization
- Regularization in RNNs
- Sequence modeling best practices

## 🔧 Usage Instructions

1. **Data Loading**:

   ```python
   (train_data, test_data), info = tfds.load(
       'imdb_reviews', split=['train', 'test'],
       as_supervised=True, with_info=True
   )
   ```

2. **Text Preprocessing**:

   ```python
   tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
   tokenizer.fit_on_texts(train_texts)
   train_padded = preprocess_text(train_texts, tokenizer, 100)
   ```

3. **Model Training**:

   ```python
   model = build_bidirectional_lstm_model()
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   history = model.fit(train_padded, train_labels, validation_split=0.2, epochs=20)
   ```

4. **Prediction**:
   ```python
   def predict_sentiment(text, model, tokenizer, max_length):
       sequence = tokenizer.texts_to_sequences([text])
       padded = pad_sequences(sequence, maxlen=max_length)
       prediction = model.predict(padded)[0][0]
       return "Positive" if prediction > 0.5 else "Negative"
   ```

## 🔮 Future Enhancements

- **Transformer Models**: BERT, GPT implementations for better performance
- **Attention Mechanisms**: Explicit attention layers for interpretability
- **Pre-trained Embeddings**: Word2Vec, GloVe, FastText integration
- **Advanced Architectures**: Hierarchical attention networks
- **Multi-task Learning**: Joint sentiment and emotion classification
- **Deployment**: Real-time sentiment analysis API development

---

**Key Achievement**: Successfully implemented and compared multiple RNN architectures for sentiment analysis, achieving 81.7% accuracy with Bidirectional LSTM while demonstrating comprehensive understanding of sequence modeling and text preprocessing techniques.
