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

    Data Preprocessing:
        The dataset was preprocessed to clean and tokenize tweets, preparing them for further analysis. Tokens were generated to facilitate text vectorization.

    Vectorization:
        Vectorization was performed using 2 types of vectorizers, CountVectorizer and TfidfVectorizer. The vocabulary size derived from a preliminary BERT tokenizer training session was used to determine the max_features parameter for vectorization.

    Balancing the Dataset:
        Due to significant class imbalance, the dataset was undersampled to ensure equal representation of both classes (hate speech and non-hate speech) during model training.

    Model Training and Evaluation:

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
