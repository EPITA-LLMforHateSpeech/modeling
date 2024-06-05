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
