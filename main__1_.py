#!/usr/bin/env python
# coding: utf-8

# In[76]:


from tensorflow import keras
# keras module for building LSTM 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 


import pandas as pd
import numpy as np
import string, os 

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[77]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Enabled")
    except RuntimeError as e:
        print(e)


# In[79]:


import re
curr_dir = 'data'
all_goals = []
for filename in os.listdir(curr_dir):
    try:
        objFile = open(os.path.join(curr_dir,filename), "r")
        fileContent = objFile.read()
        start = fileContent.find("GOAL") + len("GOAL")
        end = fileContent.find("== DATA") or fileContent.find("==== DATA")
        substring = fileContent[start:end]
        all_goals.append((substring.replace('\n','').strip('==')))
    except:
        pass
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 

corpus = [clean_text(x) for x in all_goals]
corpus[:10]


# In[80]:


tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
inp_sequences[:10]


# In[81]:


max_sequence_len = max([len(x) for x in inp_sequences])
input_sequences = np.array(pad_sequences(inp_sequences, maxlen=max_sequence_len, padding='pre'))
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label =  tf.keras.utils.to_categorical(label, num_classes=total_words)
print(max_sequence_len)
print(input_sequences)


# In[82]:


def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()
    
    # Add Input Embedding Layer
    model.add(Embedding(total_words, 10, input_length=input_len))
    
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    
    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

model = create_model(max_sequence_len, total_words)
model.summary()


# In[83]:


model.fit(predictors, label, epochs=100, verbose=5)


# In[103]:


# Saving the model:
#model.save('unsupervised_model')

champion_model = tf.keras.models.load_model('unsupervised_model')


# In[104]:


def text_generator(starting_text,number_of_words,model,max_sequence_len):
    for _ in range(number_of_words):
        token_list = tokenizer.texts_to_sequences([starting_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predict_x=model.predict(token_list, verbose=0) 
        predicted=np.argmax(predict_x,axis=1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        starting_text += ' ' + output_word
    return starting_text


# In[106]:


prompt = 'Enter the start of the text generation?\n'
i=input(prompt)


# In[109]:


text_generator(i,200,champion_model,max_sequence_len)


# In[110]:





# In[ ]:




