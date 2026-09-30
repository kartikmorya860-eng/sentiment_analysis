# Sentiment Analysis with SimpleRNN

A compact deep learning project that demonstrates sentiment classification using a Simple Recurrent Neural Network (SimpleRNN) built with TensorFlow/Keras. The project uses the IMDb movie reviews dataset to classify text as positive or negative sentiment.

## Overview

This notebook walks through the typical steps of a text classification pipeline:

- Loading and preparing the IMDb dataset
- Tokenizing text into integer sequences
- Padding sequences to a fixed length
- Building an embedding layer and SimpleRNN model
- Compiling and training the model
- Evaluating sentiment predictions

The goal is to provide a clear and practical example of how recurrent neural networks can be applied to sentiment analysis tasks.

## Project Structure

```text
sentiment_analysis/
├── sentiment_analysis_simplernn.ipynb   # Main notebook with model workflow
├── README.md                             # Project documentation
```

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Jupyter Notebook

## Model Architecture

The model follows a basic sequential design:

1. Embedding layer
   - Converts integer token IDs into dense vector representations
2. SimpleRNN layer
   - Learns patterns from sequential word data
3. Dense output layer
   - Uses a sigmoid activation for binary classification

A representative architecture used in the notebook is:

```python
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=2, input_length=50))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
```

## Training Setup

The workflow uses:

- IMDb dataset (`keras.datasets.imdb`)
- `num_words=10000` to restrict vocabulary size
- `pad_sequences(..., padding='post', maxlen=50)` to normalize input length
- Binary cross-entropy loss for classification
- Adam optimizer
- Accuracy as the evaluation metric

Example training configuration:

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd sentiment_analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install tensorflow jupyter numpy
```

### 4. Launch the notebook

```bash
jupyter notebook
```

Then open `sentiment_analysis_simplernn.ipynb` and run each cell in sequence.

## How It Works

1. The IMDb dataset is loaded.
2. Reviews are converted into token sequences.
3. Padding makes all inputs the same length.
4. Embedding layers transform tokens into meaningful vector representations.
5. The SimpleRNN processes word order and context across the sequence.
6. A sigmoid output produces a probability for positive sentiment.

## Use Cases

This project is useful for understanding:

- Text classification fundamentals
- Recurrent neural networks for NLP
- Tokenization and sequence preprocessing
- Embedding-based learning for sentiment analysis

## Challenges and Considerations

- SimpleRNNs are less powerful than modern transformer models for long text sequences.
- Vocabulary size and sequence length strongly affect training performance.
- Larger datasets and more epochs usually improve accuracy.
- Hyperparameter tuning may be needed for better results.

## Future Improvements

Potential enhancements include:

- Adding a validation and training accuracy plot
- Using LSTM or GRU for better sequence modeling
- Preprocessing with stopword removal and lemmatization
- Adding model checkpointing and early stopping
- Evaluating on custom text inputs outside the IMDb dataset
- Experimenting with word embeddings such as GloVe or pretrained transformer models

## License

This project is intended for educational and learning purposes.

## Author

Kartik 

Developed as a practical deep learning example for sentiment analysis using a recurrent neural network.
