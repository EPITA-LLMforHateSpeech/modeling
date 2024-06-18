# Modeling Tasks

## Data

The data are stored as a CSV and as a pickled pandas dataframe (Python 2.7), both stored under the `./data/` folder. Please clone the [Davidson repository](https://github.com/t-davidson/hate-speech-and-offensive-language) and move the dataset files (either in .csv or .p) to the `./data/` folder.

Each data file contains 5 columns:

`count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

`hate_speech` = number of CF users who judged the tweet to be hate speech.

`offensive_language` = number of CF users who judged the tweet to be offensive.

`neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

`class` = class label for majority of CF users.
0 - hate speech
1 - offensive language
2 - neither

For this project, we will be using the dataset as a binary classification dataset with `class 0` as `1` for hate speech and `class 1` and `class 2` and `0` for hate speech.

## Preprocessing

This repository contains code for preprocessing text data for both general natural language processing (NLP) tasks and specifically for BERT-based models.

### General NLP Preprocessing

The general NLP preprocessing steps include the following:

1. **Lowercasing:** Convert all text to lowercase to ensure uniformity.
2. **HTML Entity Unescaping:** Replace HTML entities with their corresponding characters.
3. **Removing RT Markers:** Remove "RT" markers, typically indicating retweets.
4. **Removing User Mentions:** Remove mentions of Twitter handles (@username).
5. **Removing URLs:** Remove URLs from the text.
6. **Removing Special Characters:** Remove any special characters from the text.
7. **Removing Numeric Digits:** Remove numeric digits from the text.
8. **Tokenization:** Tokenize the text into individual words or tokens.
9. **Removing Stopwords:** Remove common stopwords (e.g., 'the', 'and', 'is') from the text.
10. **Lemmatization:** Lemmatize words to reduce them to their base or dictionary form.

### BERT Preprocessing

For BERT-based models, the preprocessing steps are similar but with some specific considerations for BERT tokenization:

1. **Lowercasing:** Convert all text to lowercase.
2. **HTML Entity Unescaping:** Replace HTML entities with their corresponding characters.
3. **Removing User Mentions:** Remove mentions of Twitter handles (@username).
4. **Removing URLs:** Remove URLs from the text.
5. **Removing Special Characters:** Remove any special characters.
6. **Removing Numeric Digits:** Remove numeric digits.
7. **Tokenization:** Tokenize the text into WordPieces using BERT's tokenizer.
8. **Removing Stopwords:** BERT's tokenizer does not explicitly handle stopwords, so they are retained.
9. **Lemmatization:** Not applicable for BERT, as WordPieces are already subword units.


### Output Files

After preprocessing, the preprocessed data is saved in the following files under the `preprocessed` folder:

- For BERT-based models: `tweets_bert.txt` (text format) and `tweets_bert.pkl` (pickle format).
- For other NLP tasks: `tweets_general.csv`.

## Modeling


### Using Logistic Regression
#### Overview

This section of the project focuses on detecting hate speech in tweets using traditional machine learning techniques, specifically logistic regression. The dataset used is preprocessed and tokenized for efficient modeling. This section of the README provides an overview of the steps followed in the notebook titled `logistic_regression`.
Steps

**Data Preprocessing:**
    The dataset was preprocessed to clean and tokenize tweets, preparing them for further analysis. Tokens were generated to facilitate text vectorization.

**Vectorization:**
    Vectorization was performed using 2 types of vectorizers, CountVectorizer and TfidfVectorizer. The vocabulary size derived from a preliminary BERT tokenizer training session was used to determine the max_features parameter for vectorization.

**Balancing the Dataset:**
    Due to significant class imbalance, the dataset was undersampled to ensure equal representation of both classes (hate speech and non-hate speech) during model training.

**Model Training and Evaluation:**
    Two logistic regression models were trained and evaluated on two types of vectorized datasets (CountVectorizer and TfidfVectorizer). The performance metrics of each model were compared.
    
    Results:
        CountVectorizer:
                Accuracy: 77.97%
                Precision: 75.44%
                Recall: 78.81%
                F1 Score: 77.09%
                ROC AUC: 78.02%
            TfidfVectorizer (superior performance):
                Accuracy: 79.20%
                Precision: 75.69%
                Recall: 82.16%
                F1 Score: 78.79%
                ROC AUC: 79.36%

Based on these metrics, TfidfVectorizer was identified as the preferred vectorizer due to its higher recall, F1 score, and ROC AUC, indicating better hate speech detection capabilities.

#### Conclusion

The logistic regression models trained on tfidf vectorized datasets showed better results in hate speech detection from tweets. Future work may involve experimenting with other advanced models or fine-tuning hyperparameters to further improve performance.

### CNN Model

#### Data Preparation

- Used the same preprocessed CSV containing tweets and labels.
- Trained a custom tokenizer on the entire tweet dataset to convert text to sequences.

#### Training on Entire Dataset

- Trained the CNN model on the entire dataset for 3 epochs.
- Results:
```
  - Epoch 1: Accuracy: 0.9410, Loss: 0.2151, Val Accuracy: 0.9431, Val Loss: 0.1527
  - Epoch 2: Accuracy: 0.9480, Loss: 0.1264, Val Accuracy: 0.9405, Val Loss: 0.1794
  - Epoch 3: Accuracy: 0.9789, Loss: 0.0527, Val Accuracy: 0.9187, Val Loss: 0.2529
```
#### Training on Balanced Dataset

- Balanced the dataset by undersampling the majority class.
- Trained the CNN model on the balanced dataset for 3 epochs.
- Results:
```
  - Epoch 1: Accuracy: 0.9179, Loss: 0.2408, Val Accuracy: 0.8951, Val Loss: 0.2497
  - Epoch 2: Accuracy: 0.9802, Loss: 0.0643, Val Accuracy: 0.9178, Val Loss: 0.2373
  - Epoch 3: Accuracy: 0.9960, Loss: 0.0232, Val Accuracy: 0.9231, Val Loss: 0.2608
```
#### Performance Comparison

- The CNN model trained on the balanced dataset performed better and showed more stable validation metrics compared to the logistic regression models.
- The balanced CNN model demonstrated improved generalization and higher validation accuracy.

#### Conclusion

Training a CNN model on the balanced dataset resulted in significantly higher accuracy (99.60% training, 92.31% validation) compared to the logistic regression model, which achieved approximately 79.20% accuracy on the same dataset.

### Pretrained BERT

- **Data Preparation**: The dataset was split into training and validation sets using `train_test_split`. Token sequences were converted to tensors and padded for efficient batch processing.
  There were no steps taken to balance the dataset for this model.

- **Model Setup**: Utilized `BertForSequenceClassification` from Hugging Face's Transformers library, initialized with `bert-base-uncased` for pre-trained weights.

- **Training Process**:
  - **Optimizer**: AdamW optimizer with a learning rate of `1e-5`.
  - **Loss Function**: Cross-entropy loss (`CrossEntropyLoss`) used for training.
  - **Epochs**: Trained over 3 epochs, tracking average loss per epoch.
  - **Results**:
  ```
    - Epoch 1: Average loss of `0.2146`
    - Epoch 2: Average loss of `0.1565`
    - Epoch 3: Average loss of `0.1341`
  ```
  
#### Comparison with Logistic Regression and CNN Models

- **BERT vs Logistic Regression**:

  - BERT achieved a validation accuracy of 94%, significantly higher than logistic regression's accuracy of 79.20%.

- **BERT vs CNN**:
  - BERT and CNN both achieved high validation accuracies (94% and 92.31%, respectively). BERT performed slightly better considering the dataset it was trained on was not balanced.

### Key points

- Training BERT on the tweet dataset for hate speech classification resulted in superior performance compared to logistic regression and comparable performance to CNN models.
- Both logistic regression and CNN models required dataset balancing techniques to mitigate overfitting and improve generalization, which was not the case with the pretrained BERT model.
- BERT achieved a validation accuracy of 94%, and CNN 92.31%.
- The training of BERT took approximately 7 hours over 3 epochs.
- The comparison with CNN is based on accuracy alone, as recall and precision for the CNN model were not measured in this evaluation.


# BiLSTM Hate Speech Detection Model

This repository contains code for training a Bidirectional LSTM (BiLSTM) model to detect hate speech in tweets. The model is built using TensorFlow and Keras, and it is trained on a preprocessed dataset of tweets.

## Introduction
This project aims to create a machine learning model that can identify hate speech in text data, specifically tweets. The model uses a Bidirectional LSTM (BiLSTM) network to process and classify the tweets.

## Data Preparation
1. **Load the Dataset:** The dataset is loaded from a preprocessed pickle file containing tweets and their corresponding labels.
2. **Preprocess the Data:** Tokenize the tweets and pad the sequences to ensure uniform input length for the model.
3. **Split the Data:** The data is split into training and testing sets to evaluate the model's performance.

## Model Architecture
The model is defined using a Sequential API with the following layers:
- **Embedding Layer:** Converts the input tokens into dense vectors of fixed size.
- **Bidirectional LSTM Layers:** Two Bidirectional LSTM layers to capture dependencies in both directions.
- **Dropout Layers:** Applied after each LSTM layer to prevent overfitting.
- **Dense Layer:** A single neuron with a sigmoid activation function for binary classification.

## Training the Model
The model is compiled with Adam optimizer and binary crossentropy loss. It is trained for 20 epochs with a batch size of 128, and a validation split of 20% to monitor performance on unseen data.

## Evaluation
After training, the model's performance is evaluated on the test set. Key metrics include accuracy, precision, recall, and F1-score. The model achieves an overall test accuracy of approximately 92%.

## Saving the Model
The trained model is saved in HDF5 format for future use. The HDF5 format is chosen for its compatibility and ease of use.

## Usage
To use the trained model for prediction, load the model and preprocess the input text in the same manner as the training data. Pass the preprocessed text to the model to obtain predictions.

## Summary after Training
The model summary provides details about the layers, their output shapes, and the number of parameters. The final model has around 2.34 million parameters, with approximately 780,000 trainable parameters.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code with proper attribution.

---

This README provides an overview of the project, detailing the steps from data preparation to model training and evaluation. For more detailed instructions and the actual code, refer to the source files in the repository.
