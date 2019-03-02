# Transforma at SemEval-2019 Task 6: Identifying and Categorising Offensive Language in Social Media 

This repository outlines the process that we took to tackle the OffensEval 2019 challenge. https://competitions.codalab.org/competitions/20011#learn_the_details

## Code
You need to download pre-trained word embeddings. We have used Glove Twitter which can be found here: https://nlp.stanford.edu/projects/glove/. Once you have downloaded the embeddings, you can run```python model.py```, which outputs the classification report for our BiLSTM-CNN model.

## Our approach
Our approach involved splitting this challenge into two parts: data processing and sampling and choosing the optimal deep learning architecture. Given that our datasets are unstructured and informal text data from social medias, we decided to spend more time creating our text preprocessing pipeline to ensure that we are feeding in high quality data to our model. We also experimented with two techniques, SMOTE and Class Weights to counter the imbalance between classes. Once we are happy with the quality of our input data, we proceed to choosing the optimal deep learning architecture for this task. Given the quality and quantity of data we have been given, we found that the addition of CNN layer provides very little to no additional improvement to our model's performance and sometimes even worsen our F1-score. **In the end, the deep learning architecture that gives us the highest macro F1-score is a simple BiLSTM-CNN.**

## Full report
The full report can be found in the Transforma-OffensEval.pdf file.