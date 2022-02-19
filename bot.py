import numpy as np

from utils import *
import joblib
import json


class Bot:
    def __init__(self):
        self.clf = joblib.load('rf.joblib')
        self.tag_list = np.load('tags.npy',allow_pickle=True)
        self.all_words = np.load('all_words.npy',allow_pickle=True)
        with open('train_intent.json', 'r') as file:
            self.intents = json.load(file)

    def __call__(self):
        print('Bot: Hello how can I help you !!')
        while True:
            inp = input('You: ')
            if inp.lower() == 'quit':
                print('Thank you for your time !!')
                break

            predict = self.clf.predict_proba([bow(inp,self.all_words)])
            predict_tag = self.tag_list[np.argmax(predict)]


            if predict[0][np.argmax(predict)] > 0.5:
                for i in self.intents['intents']:
                    if i['tag'] == predict_tag:
                        response = i['response']
                print('Bot:',response)

            else:
                print('Bot: Please wait a human operator will contact you !')

if __name__ == '__main__':
    Bot()()
