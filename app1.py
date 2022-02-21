import streamlit as st
import joblib
import string
import pickle
import re
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize




def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


model = joblib.load(open("emotion_detection_model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

emotions_emoji_dict = {4:"😠",6:"🤮", 2:"😨😱", 0:"😂", 5:"😐", 1:"😔", 7:"😳", 3:"😮"}


st.title("Emotion Detector App")

text = st.text_area("Type Here")

if st.button('Predict'):

    # 1. preprocess
    lower_text = convert_to_lower(text)
    remove_numbers_text = remove_numbers(lower_text)
    lem_text = lemmatizing(remove_numbers_text)
    rempunc_text = remove_punctuation(lem_text)
    stopword_text = remove_stopwords(rempunc_text)
    ws_text = remove_extra_white_spaces(stopword_text)

    # 2. vectorize
    vector_input = tfidf.transform([ws_text])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        st.header("{}:{}".format("Joy",emotions_emoji_dict))
    elif result == 1:
        st.header("{}:{}".format("Sadness",emotions_emoji_dict))
    elif result == 2:
        st.header("{}:{}".format("Fear",emotions_emoji_dict))
    elif result == 3:
        st.header("{}:{}".format("Surprise",emotions_emoji_dict))
    elif result == 4:
        st.header("{}:{}".format("Anger",emotions_emoji_dict))
    elif result == 5:
        st.header("{}:{}".format("Neutral",emotions_emoji_dict))
    elif result == 6:
        st.header("{}:{}".format("Disgust",emotions_emoji_dict))
    else:
        st.header("{}:{}".format("Shame",emotions_emoji_dict))