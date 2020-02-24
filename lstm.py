#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:54:31 2020

@author: seangao
"""

import json
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from sklearn.model_selection import train_test_split


#LOAD DATA
with open('/Users/seangao/Desktop/Research/lstmYelp/yelp_dataset/review.json') as f:
    reviews = [json.loads(line) for line in f]
    
df = pd.DataFrame(reviews)

df1 = df[['text','stars']]

#PRE-PROCESS
df1['sentiment'] = np.where(df1['stars'] >=3, 'pos', 'neg')
df1['text'] = df1['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#VECTORIZE TEXT
tokenizer = Tokenizer(nb_words=2500, lower=True,split=' ')
tokenizer.fit_on_texts(df1['text'].values)

X = tokenizer.texts_to_sequences(df1['text'].values)
X = pad_sequences(X)

#BUILD MODEL
embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(Embedding(2500, embed_dim,input_length = X.shape[1], dropout = 0.2))
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
print(model.summary())

#TRAIN MODEL
y = pd.get_dummies(df1['sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, 
                                                      random_state = 42)

model.fit(X_train, y_train, batch_size = batch_size, 
          epochs = 1,  verbose = 0)

#CHECK ACCURACY
score, acc = model.evaluate(X_test, y_test, 
                            batch_size = batch_size, verbose = 0)