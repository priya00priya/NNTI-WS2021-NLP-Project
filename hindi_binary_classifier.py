# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:35:41 2021

@author: piyab
"""
import os
#os.chdir("D:/Saarland/NN TI/NNTI_WS2021_Project")

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


import pandas as pd
import TASK_1.py
from Task1_word_Embeddings.ipynb import *
SEED = 1234

#torch.manual_seed(SEED)
#torch.backends.cudnn.deterministic = True

#TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm')
#LABEL = data.LabelField(dtype = torch.float)


df = pd.DataFrame.from_csv("hindi_hatespeech.tsv", sep="\t")

X = df['text']
y = df['task_1']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

def preprocessing(input_data):
    dataset = pd.DataFrame(input_data)
    dataset['text'] = dataset['text'].str.lower()
    
    # data preprocessing
    dataset['text'] = dataset['text'].map(lambda x: clean_data(x))
    # drop empty values 
    dataset = drop_empty_values(df)
    
    #building vocabulary
    sentences, all_words, v = building_vocabulary(df)
    
    #Calculating word frequencies
    frequency_of_words = calculating_word_frequency(all_words)
    
    return dataset

class Attention(torch.nn.Module):
	def __init__(self, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(Attention, self).__init__()

		self.output_size = output_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.embedding_length = embedding_length
		
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weights = nn.Parameter(weights, requires_grad=False)
		self.lstm = nn.LSTM(embedding_length, hidden_size)
		self.label = nn.Linear(hidden_size, output_size)
		#self.attn_fc_layer = nn.Linear()
		
	def Attention_Net(self, lstm_output, final_state):
		
		hidden = final_state.squeeze(0)
		attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
		soft_attn_weights = F.softmax(attn_weights, 1)
		new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
		
		return new_hidden_state
	
	def forward(self, input_sentences, batch_size=None):
	
		input = self.word_embeddings(input_sentences)
		input = input.permute(1, 0, 2)
		if batch_size is None:
			h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
			c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())
			
		output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0)) 
		output = output.permute(1, 0, 2)
		
		attn_output = self.Attention_Net(output, final_hidden_state)
		logits = self.label(attn_output)
		
		return logits

X_train = preprocessing(X_train)
'''
INPUT_DIM = len(v)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256 #Size of the hidden_state of the LSTM
OUTPUT_DIM = 1
'''
out_w = io.open('embedding_weight_W.tsv', 'w', encoding='utf-8')
out_w1 = io.open('embedding_weight_W1.tsv', 'w', encoding='utf-8')

weight_w = []
for x in out_w:
     words = [x for x in line.split(',')]
     weight_w.append(words)
    
weight_w1 = []
for x in out_w1:
    words = [x for x in line.split(',')]
    weight_w1.append(words)


model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)