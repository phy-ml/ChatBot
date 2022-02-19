import json
from utils import *
import random
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# get the train intents
with open('train_intent.json','r') as file:
    intents = json.load(file)

# get the test intents
with open('test_intent.json','r') as file:
    test_intent = json.load(file)

all_words = []
tags = []
document = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['pattern']:
        w = tokenize(pattern)
        all_words.extend(w)
        document.append((w,tag))

ignore_word = ['?','.','!','@',',','&','(',')','*']
all_words = [lemm(i) for i in all_words if i not in ignore_word]
all_words = sorted(list(set(all_words)))
all_words = [i for i in all_words if only_letters(i)]
tags = sorted(set(tags))
# print(all_words)

####################################################################
training = []
for doc in document:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemm(i) for i in pattern_words]

    for i in all_words:
        bag.append(1) if i in pattern_words else bag.append(0)

    training.append([bag,doc[1]])

random.Random(0).shuffle(training)

training = np.array(training)

###########################################################################
# clean the test dataset and create a bag of words for it with associated tags
testing = []
for intent in test_intent['intents']:
    tag = intent['tag']
    for pattern in intent['pattern']:
        # print(pattern, tag)
        # print(bow(pattern,all_words))
        testing.append([bow(sentence=pattern,all_words=all_words),tag])

random.Random(0).shuffle(testing)

#########################################################################
# save the train and test data
# with open('train.npy','wb') as f:
#     np.save(f,training)
#
# with open('test.npy','wb') as f:
#     np.save(f,testing)

# save the list of all words created
with open('all_words.npy','wb') as f:
    np.save(f,all_words)

# save all the tag created
with open('tags.npy','wb') as f:
    np.save(f,tags)
##########################################################################
# train the model with random forest
# seperate the train and test dataset into x and y components
train_x = np.array(list(training[:,0]))
train_y = training[:,1]

# print(np.array(list(np.array(testing)[:,0])))

test_x = np.array(list(np.array(testing)[:,0]))
test_y = np.array(list(np.array(testing)[:,1]))

# encode the targets
label = LabelEncoder()
en_train_y = label.fit_transform(train_y)
en_test_y = label.fit_transform(test_y)

clf = RandomForestClassifier()
clf.fit(np.array(list(train_x)),np.array(en_train_y))

# save the model
joblib.dump(clf,'rf.joblib')

###########################################################################

print('Model Training Completed')