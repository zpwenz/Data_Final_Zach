from flask import Flask, render_template, redirect, request
import pandas as pd
import pickle
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
from absolang import absolutist, absolutist_index
import re, string
import nltk
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from bs4 import BeautifulSoup
import unicodedata
import spacy
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 



# Use pickle to load in the pre-trained model.
with open('dummy_model_2.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as v:
    vectorizer = pickle.load(v)

def personal_pronouns(text1):
    pronouns = ["i", "me", "my", "mine", "myself"]
    count = 0
    words = text1.split ()
    for w in words:
        if w.lower() in pronouns:
             count += 1
    return(count)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

import contractions
from contraction_list import CONTRACTION_MAP

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

app = Flask(__name__)

# Main home page
@app.route("/", methods=['GET', 'POST'])
def whatever():
    if request.method == 'GET':
        return render_template('index.html')


    if request.method == 'POST':
        #temperature = request.form['temperature']
        text = request.form['text']
        textc = replace_contractions(text)
        text1 = [textc]
        text2 = to_lowercase(text1)
        text3 = str(text2)

        #windspeed = request.form['windspeed']
        #input_variables = pd.DataFrame([[text]],
                                       #columns=['Text']) 
        #prediction = model.predict(input_variables)[0]
        inputvector =  vectorizer.transform(text1)
        prediction = model.predict_proba(inputvector)[0][1]
        absolutist = absolutist_index(text3)
        sentiments = SentimentIntensityAnalyzer().polarity_scores(text3).get('compound')
        pronoun = personal_pronouns(text3)
        return render_template('index.html',
                                     original_input={'Text':text3},
                                     words = textc,
                                     result=prediction, absolutist = absolutist, sentiment = sentiments, pronouns = pronoun
                                     )




if __name__ == "__main__":
    app.run(debug=True)