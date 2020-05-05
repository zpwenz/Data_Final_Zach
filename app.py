import os
from flask import Flask, render_template, redirect, request
import pandas as pd
import pickle
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import numpy, textblob, string
#from keras.preprocessing import text, sequence
#from keras import layers, models, optimizers
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
from absolang import absolutist, absolutist_index
import re, string
import inflect
from nltk import word_tokenize, sent_tokenize
import unicodedata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
import json
import statistics
import numpy as np

afinn_wl_url = ('https://raw.githubusercontent.com'
                '/fnielsen/afinn/master/afinn/data/AFINN-111.txt')
sentiments = pd.read_csv(afinn_wl_url,
                          header=None, # no column names
                          sep='\t',  # tab sepeated
                          names=['term', 'value']) #new column names



# Use pickle to load in the pre-trained model.
with open('linear_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as v:
    scaler = pickle.load(v)

def personal_pronouns(text1):
    pronouns = ["i", "me", "my", "mine", "myself"]
    count = 0
    words = text1.split ()
    for w in words:
        if w.lower() in pronouns:
             count += 1
    return(count)

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

import contractions
from contraction_list import CONTRACTION_MAP

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

fatigue_list = ['fatigued', 'weary', 'bored', 'tired', 'tiring', 'frustrated', 'irritated', 'annoyed', 'antsy', 'tiresome', 'impatient', 'exhausted', 'jaded', 'sick', 'lazy', 'knackered', 'winded', 'restless', 'accustomed', 'exasperated', 'stale', 'cranky', 'discombobulated', 'dizzy', 'grumpy', 'lethargic', 'disgusted', 'scared', 'disheartened', 'wearying', 'confused', 'mad', 'emotionally_drained', 'Exhausted', 'flustered', 'nauseated', 'disillusioned', 'jet_lagged', 'disenchanted', 'frazzled', 'pooped', 'fond', 'angry', 'groggy', 'numb', 'nervous', 'bewildered', 'haggard', 'complaining', 'peeved', 'grouchy', 'anxious', 'boring', 'demoralized', 'nauseous', 'crabby', 'bummed', 'bothered', 'despondent', 'irritable', 'psyched', 'discouraged', 'dehydrated', 'Weary', 'dreading', 'overworked', 'worried', 'dejected', 'pissed', 'embarrassed', 'whining', 'angrier', 'afraid', 'rusty', 'crazy', 'ashamed', 'anymore', 'exhausting', 'feeling', 'excuses', 'monotonous', 'distracted', 'dazed', 'ragged', 'sleep_deprived', 'homesick', 'stupid', 'silly', 'frustrating', 'dispirited', 'downcast', 'hungover', 'apathetic', "'m", 'lackadaisical', 'disoriented', 'agitated', 'wishy_washy', 'mentally', 'wearily']
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

def sentiment_score(text): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    sentences = nltk.tokenize.sent_tokenize(text)
    scores = []
    for sentence in sentences:
        scores.append((sid_obj.polarity_scores(sentence)).get('compound'))

    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    return(statistics.mean(scores))
def pronomialization(text):
    noun_count = check_pos_tag(text, 'noun')
    return(personal_pronouns(text) / noun_count)

def fatigue_ratio(text):
    counts = []
    for word in fatigue_list:
        counts.append(text.count(word))
    return(sum(counts)/len(text.split()))

def get_features(text):
    S = sentiment_score(text)
    P = pronomialization(text)
    A = absolutist_index(text)
    F = fatigue_ratio(text)
    return(np.array([S, P, A, F]))

def tfs_dict(text):
    a_list = nltk.tokenize.sent_tokenize(text)
    sall = []
    for sentence in a_list:
        term = "I" #term we want to search for
        words = sentence.split() #split the sentence into individual words
        if term in words:
            replace_contractions(sentence)
            doc=nlp(sentence)
            sall.append(doc)
    verbs = []
    for sentence in sall:
        s_tokens = [token.lower_ for token in sentence]
        verbs.append(s_tokens)
    verb_list = [item for sublist in verbs for item in sublist]
    c = Counter(verb_list)
    words_f = [([w, c.get(w, 0)]) for w in sentiments["term"].tolist()]
    s_words = list(filter(lambda x: x[1] != 0, words_f))
    tf = pd.DataFrame(s_words, columns = ["term", "frequency"])
    tfv = tf.merge(sentiments)
    colors = []
    for value in tfv['value']:
        if value > 0:
            colors.append("positive")
        else:
            colors.append("negative")
    tfv["group"] = colors
    return tfv.to_dict('records')

app = Flask(__name__)

# Main home page
@app.route("/", methods=['GET', 'POST'])
def whatever():
    if request.method == 'GET':
        return render_template('index.html')


    if request.method == 'POST':
    
        text = request.form['text']
        textc = replace_contractions(text)
        text1 = [textc]
        text2 = to_lowercase(text1)
        text3 = str(text2)
        inputvector =  get_features(textc)
        inputvector = np.array(inputvector.reshape(1,-1))
        inputvector = scaler.transform(inputvector)
        prediction = model.predict_proba(inputvector)[0][1]
        absolutist = absolutist_index(text3)
        sentiments = sentiment_score(text)
        fatigue = fatigue_ratio(text)
        pronoun = pronomialization(text3)
        return render_template('index.html',
                                     original_input={'Text':text3},
                                     words = textc,
                                     result=prediction, absolutist = absolutist, sentiment = sentiments, pronouns = pronoun, fatigue = fatigue,
                                     data = json.dumps(tfs_dict(text))
                                     )
        

if __name__ == "__main__":
    app.run(debug=True)