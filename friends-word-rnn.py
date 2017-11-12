
# coding: utf-8

# In[1]:


from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import os
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from datetime import datetime


# In[ ]:


with open("./data/friends/all_scripts.txt") as scripts_fileobj:
    all_scripts = scripts_fileobj.read().strip().lower().decode('utf8').encode('ascii', errors='ignore')


# In[ ]:


def preprocess_text(text):
    lines = text.split("\n")
    dialogues = []
    dialogue = []
    for line in lines:
        if ":" in line:
            dialogues.append(" ".join(dialogue))
            dialogue = []
            dialogue.append(line)
        else:
            dialogue.append(line)
    
    text = "\n".join(dialogues)
    punctuations = set(re.findall(r"[^a-zA-Z0-9 ]",text))
    for punctuation in punctuations:
        if punctuation == "\n":
            text = text.replace(punctuation," NEWLINE ")
        else:
            text = text.replace(punctuation," "+punctuation+" ")
            
        
    return text


# In[ ]:


def remove_infrequent_tokens(tokens,min_count=10):
    word_count = {}
    new_tokens = []
    vocab = []
    for token in tokens:
        if token in word_count:
            word_count[token] +=1
        else:
            word_count[token]=1
    for token_word in tokens:
        if word_count[token_word]>min_count:
            new_tokens.append(token_word)
    return new_tokens


# In[ ]:


def get_latest_train(model_path):
    checkpoints = os.listdir(model_path)
    latest_checkpoint = ""
    highest_epoch = 0
    for checkpoint in checkpoints:
        current_epoch = int(re.findall("weights-improvement-(\d+)-(\d+\.\d+).+",checkpoint)[0][0])
        if highest_epoch < current_epoch:
            highest_epoch = current_epoch
            latest_checkpoint = checkpoint
    return latest_checkpoint,highest_epoch


# In[ ]:


all_scripts_cleaned = preprocess_text(all_scripts)
tokens = all_scripts_cleaned.split()


# In[ ]:


# print(len(tokens))
cleaned_tokens = remove_infrequent_tokens(tokens)
print("Number of cleaned tokens: {}".format(len(cleaned_tokens)))


# In[ ]:


vocab = list(set(cleaned_tokens))
print("Vocal length: {}".format(len(vocab)))


# In[ ]:


vocab_length = len(vocab)
characters2id = dict((c, i) for i, c in enumerate(vocab))
id2characters = dict((i, c) for i, c in enumerate(vocab))
section_length = 20
step = 2
sections = []
section_labels = []
for i in range(0,len(cleaned_tokens)-section_length-1,step):
    section_in = cleaned_tokens[i:i+section_length]
    section_out = cleaned_tokens[i+section_length]
    sections.append([characters2id[word] for word in section_in])
    section_labels.append(characters2id[section_out])

print("Number of training examples: {}".format(len(sections)))


# In[ ]:


X = np.reshape(sections, (len(sections), section_length, 1))
print(X.shape)
X = X / float(vocab_length)
y = np.zeros((len(sections),vocab_length))
for i,section in enumerate(sections):
    y[i,section_labels[i]] = 1


# In[ ]:


print(X.shape,y.shape)


# In[ ]:


model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


model_path = "./model/model3_friends/"
filepath="./model/model3_friends/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]


# In[ ]:


# Training
num_epochs = 40
checkpoint_filename,latest_epoch = get_latest_train(model_path)
train_filepath = model_path+checkpoint_filename
model.load_weights(train_filepath)
print("{} - Training resuming from epoch {}. Training for {} epochs.".format(datetime.now(),latest_epoch,num_epochs))
model.fit(X, y, epochs=latest_epoch+num_epochs, batch_size=128, callbacks=callbacks_list,initial_epoch=latest_epoch)
print("{} - Training finished".format(datetime.now()))


# In[ ]:


# Testing
checkpoint_filename,latest_epoch = get_latest_train(model_path)
test_filepath = model_path+checkpoint_filename
print("Testing after {} epochs. Filename: {}".format(latest_epoch,test_filepath))
model.load_weights(test_filepath)
model.compile(loss='categorical_crossentropy', optimizer='adam')

start = np.random.randint(0, len(sections)-1)
pattern = sections[start]
print("Seed:", " ".join([id2characters[idx] for idx in pattern]).replace("NEWLINE","\n"))
predictions = []
# generate characters
for i in range(500):
#     print([id2characters[idx] for idx in pattern])    
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_length)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = id2characters[index]
    seq_in = [id2characters[value] for value in pattern]
    predictions.append(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print(" ".join(predictions).replace("NEWLINE","\n"))


# In[ ]:


print(vocab)

