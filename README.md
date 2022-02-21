# Emotion-Detection-Text-Classification-

![edss](https://user-images.githubusercontent.com/66901829/154924493-20b2f3c0-8035-4770-ae43-7d019048f281.png)

## Overview
This is a multiclass sentiment analysis problem in which given an input text, the goal is to classify it as one of 8 given emotions that best represent the mental state of the writer.

## Data Overview
The dataset was taken from Paperswithcode (has been provided in files above). We are provided with some texts and their corresponding emotions in tabular form.
There are 8 given type of emotions-
- Joy
- Sadness
- Fear
- Surprise
- Anger
- Neutral
- Disgust
- Shame

## Installation
The Code is written in Python 3.6.13 If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Overall Approach

EDA part
- We performed sentiment analysis for our emotions, that whether the emotion is positive, negative or neutral.
- We extracted most common keywords for each of the emotion 
- We generated word clouds of the most common keywords for every emotion

Class Imbalance

We had imbalance in the emotion classes with most of the texts belonging to 'Joy' class. Therefore we performed data augmentation in which we generated synthetic texts for emotion classes having low count of texts. Data augmentation was performed with the help of transformer based model BERT-base-uncased.

Model

In this project, our best performing model was Random Forest. We further tuned our model with hyperparameter optimization using RandomizedSearchCV.
