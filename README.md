# RNN-LSTM-BILSTM
# Fake News Detection using Recurrent Neural Networks

This notebook demonstrates a fake news detection system using various Recurrent Neural Network (RNN) architectures implemented with TensorFlow/Keras.

## Dataset
The project utilizes a dataset containing news articles, their titles, authors, and a `label` indicating whether the news is real (0) or fake (1).

## Steps Performed:

1.  **Data Loading and Initial Exploration:**
    *   Loads the dataset using pandas.
    *   Handles missing values by dropping rows with `NaN`.
    *   Separates independent features (X) and dependent features (y).

2.  **Text Preprocessing:**
    *   Copies the 'title' column for text processing.
    *   Uses NLTK for text cleaning:
        *   Removes non-alphabetic characters.
        *   Converts text to lowercase.
        *   Removes stopwords.
        *   Applies Porter Stemmer for word normalization.
    *   Converts processed text into one-hot representations based on a defined vocabulary size.
    *   Pads sequences to a fixed length (`sent_length=20`) for uniform input to the neural networks.

3.  **Model Building and Training:**
    *   **Model 1: Simple RNN**
        *   Architecture: `Embedding` layer -> `SimpleRNN` layer -> `Dense` output layer with sigmoid activation.
        *   Compiled with `binary_crossentropy` loss and `adam` optimizer.
        *   Trained for 10 epochs with a batch size of 64.
    *   **Model 2: LSTM**
        *   Architecture: `Embedding` layer -> `LSTM` layer -> `Dense` output layer with sigmoid activation.
        *   Compiled with `binary_crossentropy` loss and `adam` optimizer.
        *   Trained for 10 epochs with a batch size of 64.
    *   **Model 3: Bidirectional LSTM with Dropout**
        *   Architecture: `Embedding` layer -> `Bidirectional(LSTM)` with `return_sequences=True` -> `Dropout` -> `Bidirectional(LSTM)` -> `Dropout` -> `Dense` (ReLU) -> `Dense` (Sigmoid).
        *   Compiled with `binary_crossentropy` loss and `adam` optimizer.
        *   Trained for 10 epochs with a batch size of 64.

4.  **Model Evaluation:**
    *   For each model, predictions are made on the test set.
    *   Predictions are converted to binary (0 or 1) using a threshold of 0.5.
    *   `accuracy_score` is calculated to assess performance.
    *   `confusion_matrix` is generated and visualized using seaborn heatmaps to show true positives, true negatives, false positives, and false negatives.
