import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lemm(word):
    return lemmatizer.lemmatize(word.lower())

def clean_stentence(sentence):
    tok = tokenize(sentence=sentence)
    return [lemm(word=word) for word in tok]

def bow(sentence,all_words):
    sentence = clean_stentence(sentence=sentence)
    bag = [0]*len(all_words)
    for s in sentence:
        for i,w in enumerate(all_words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def only_letters(word):
    letters = set('abcdefghijklmnopqrstuvwxyz')
    for char in word.lower():
        if char not in letters:
            return False
    return True

# words = ['organize','organizes','organizing']
# stemmed_words = [stem(i) for i in words]
# print(stemmed_words)