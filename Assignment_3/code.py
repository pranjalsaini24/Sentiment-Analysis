# ADD THE LIBRARIES YOU'LL NEED
import pandas as pd
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from tensorflow.keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from gensim.models import FastText
import fasttext
import fasttext.util
from keras.initializers import Constant
from keras.models import model_from_json
import tensorflow as tf
from keras.preprocessing import sequence
import numpy as np
import nltk
import json
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from keras.layers import Activation
from keras import backend as K
from gensim.models import Word2Vec
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import RandomOverSampler
#from sklearn.utils import class_weight
#from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader 
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''
#vocab_sorted=dict()
vocab_embedding=dict()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model_classification(torch.nn.Module):

  def __init__(self,vocab,hidden_dim,embedding_dim):

    super().__init__()
    self.hidden_dim=hidden_dim
    self.embedding=nn.Embedding(len(vocab)+1,embedding_dim,padding_idx=0)
    self.bigru=nn.GRU(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
    self.gru=nn.GRU(embedding_dim,hidden_dim,batch_first=True)
    self.dropout=nn.Dropout()
    self.rnn=nn.RNN(embedding_dim,hidden_dim,batch_first=True)
    self.linear=nn.Linear(2*hidden_dim,5)
    self.lstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True)
    self.bilstm=nn.LSTM(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
  
  def forward(self,x):
    h0=torch.zeros(1,x.size(0),self.hidden_dim)
    #x.type(torch.DoubleTensor)
    embedding_reviews=self.embedding(x)
    #print(embedding_reviews)
    #print(embedding_reviews.dtype)
    #embedding_reviews=embedding_reviews.type(torch.FloatTensor)
    #print(device)
    #embedding_reviews.to(device)
    #print(embedding_reviews)
    
    gru_rep,(hidden,ct)=self.bilstm(embedding_reviews)
    #print(gru_rep)
    #print(hidden[-2,:,:])
    #print(hidden[-1,:,:])
    #print(hidden2)
    hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
    #print(hidden)
    #print("hello")
    #pt=torch.cat((ht,ct),1)
    #print(ht[-1])
    pt=self.dropout(hidden)
    #print(pt)
    #gru_rep_dropout=self.dropout(ht[-1])
    #print(gru_rep_dropout)
    output=self.linear(hidden)
    return output

class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        #idx=0
        return torch.tensor(self.X[idx]).to(device), torch.tensor(self.y[idx]).to(device)


def over_sample_data(train):
  oversample = RandomOverSampler(random_state=777)
  X_over, y_over = oversample.fit_resample(train[['reviews','ratings']], train['ratings'])
  #print(len(y_over))
  #X_over,y_over=smote.fit_sample(train['reviews'],train['ratings'])
  #X_over, y_over = undersample.fit_resample(train[['reviews','ratings']], train['ratings'])
  columnnames=['reviews','ratings']
  df = pd.DataFrame(X_over, columns=columnnames)
  df['ratings'].value_counts()
  train=df
  train = train.sample(frac=1).reset_index(drop=True)
  #reviews=train['reviews']
  #ratings=train['ratings']
  return train


def pre_process_rating(data):
  ratings=data['ratings']
  ratings=ratings.values.tolist()
  #print(ratings[2],ratings[7],ratings[3])
  ratings=[x-1 for x in ratings]
  #print(ratings[2],ratings[7],ratings[3])
  return ratings

def encode_data(text):
  reviews=text
  vocab=dict()
  for elem in reviews:
    for words in elem:
      if words in vocab:
        vocab[words]+=1
      else:
        vocab[words]=1
  #vocab_sorted=vocab  
  with open('./vocab.json') as json_file: 
    vocab_sorted = json.load(json_file)

  
  
  #f=open('/content/drive/MyDrive/NLP_Assignment1/glove.twitter.27B.200d.txt')
  #for line in f:
  #  line=line.split(' ')
    #print(line[0])
  #  vocab_embedding[line[0]]=np.asarray(line[1:],dtype='float32').tolist()
  reviews1=list()
  replace_data=list()
  #print(reviews[0])
  #print(reviews[100])
  for elem in reviews:
    replace_data=list()
    for word in elem:
      if(word in vocab_sorted):
        replace_data.append(vocab_sorted[word])
    reviews1.append(replace_data)
  reviews=reviews1
  return reviews
  #len(vocab)

    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    
    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like: 
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to 
    # encode the words into integers 

    # return encoded examples



def convert_to_lower(text):
  reviews=text
  reviews1=list()
  for elem in reviews:
    reviews1.append(elem.lower())
  reviews=reviews1
  reviews_orignal=reviews
  return reviews 
    # return the reviews after convering then to lowercase


def remove_punctuation(text):
  reviews=text
  punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~+=\`'''
  reviews1=list()
  for elem in reviews:
    for char in elem:
      if(char in punc):
        elem=elem.replace(char,"")
    reviews1.append(elem)
  reviews=reviews1
  return reviews
    # return the reviews after removing punctuations


def remove_stopwords(text):
  #stop_words = set(stopwords.words('english')) 
  return text
    # return the reviews after removing the stopwords

def perform_tokenization(text):
  reviews=text
  nltk.download('punkt')
  reviews1=list()
  review=list()
  for elem in reviews:
    #print(elem)
    review=word_tokenize(elem)
    reviews1.append(review)
  reviews=reviews1
  return reviews
    # return the reviews after performing tokenization


def perform_padding(data):
  reviews=data
  review_length=29
  #print(review_length)
  reviews= sequence.pad_sequences(reviews, maxlen=review_length,padding="post")
  #print(reviews[2])
  #reviews.shape
  return reviews
    # return the reviews after padding the reviews to maximum length

def preprocess_data(data,is_train=True):
    # make all the following function calls on your data
    # EXAMPLE:->
      
        
      review = data["reviews"].values.tolist()
      review = convert_to_lower(review)
      review = remove_punctuation(review)
      review = remove_stopwords(review)
      review = perform_tokenization(review)
      review = encode_data(review)
      review = perform_padding(review)
        
      return review



def softmax_activation(res):
  maxes = torch.max(res, 1, keepdim=True)[0]
  maxes.to(device)
  #print(res-maxes)
  res_exp = torch.exp(res-maxes)
  res_exp.to(device)
  #print(res_exp)
  res_exp_sum = torch.sum(res_exp, 1, keepdim=True)
  res_exp_sum.to(device)
  #print(res_exp_sum)
  output_custom = res_exp/res_exp_sum
  output_custom.to(device)
  return output_custom
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)



class NeuralNet:

    def __init__(self, reviews, ratings):

        self.reviews = reviews
        self.ratings = ratings



    def build_nn(self):
      with open('./vocab.json') as json_file: 
        vocab_sorted = json.load(json_file)
      vocab_size=len(vocab_sorted)
      
      vocab_embedding = fasttext.load_model('./crawl-300d-2M-subword.bin')
      embedding_dim=300
      embedding_matrix=np.zeros((vocab_size+1, embedding_dim))
      i=1
      for words in vocab_sorted:
        #print(vocab_sorted['intelligent'])
        vector=vocab_embedding.get_word_vector(word=words)
        #vector=vocab.get(word)
        #if word in vocab.wv.vocab:
        #  vector=vocab[word]
        if vector is not None:
            embedding_matrix[i]=vector
        i+=1

      self.embedding_matrix=embedding_matrix
      self.model=Model_classification(vocab_sorted,50,300)
      return self.model

    def train_nn(self,batch_size,epochs,sampler):
      lr=0.001
      train_ds = ReviewsDataset(self.reviews,self.ratings)
      train_dl = DataLoader(train_ds, batch_size=128,sampler=sampler)
      
      model=self.model
      model.to(device)
      model.double()
      embedding_matrix_1=torch.tensor(self.embedding_matrix).to(device)
      ##embedding_matrix_1=embedding_matrix_1.type(torch.DoubleTensor)
      model.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix_1))
      model.embedding.weight.requires_grad = False
      parameters=filter(lambda p: p.requires_grad, model.parameters())
      #check=sum(p.numel() for p in model.parameters() if p.requires_grad)
      #print(check)
      nll_loss = nn.NLLLoss()
      optimizer = torch.optim.Adam(parameters, lr=lr)
      for epoch in range(epochs):
        
        correct=0
        total=0
        sum_loss=0
        model.train()
        for x,y in train_dl:
          #print(x)
          Y_pred=model(x).to(device)
          #break
          #print(Y_pred)
          Y_pred=softmax_activation(Y_pred)
          Y_pred.to(device)
          optimizer.zero_grad()
          #print(Y_pred)
          #print(y)
          #print(torch.log(Y_pred)[-1])
          #print(y)

          loss=nll_loss(torch.log(Y_pred),y)
          loss.to(device)
          loss.backward()
          optimizer.step()
          pred = torch.max(Y_pred, 1)[1].to(device)
          #print(pred)
          correct += (pred== y).float().sum()
          total+=y.shape[0]
          sum_loss += loss.item()*y.shape[0]
        if(epoch%1==0):
          print(epoch,"th Epoch, Train Accuracy=",(correct/total),"Train_loss=",(sum_loss/total))
            # write the training loop here; you can use either tensorflow or pytorch
            # print validation accuracy

    def predict(self, reviews):
      Y_pred=self.model(torch.tensor(reviews).to(device)).to(device)
      pred = torch.max(Y_pred, 1)[1]
    
      #test_reviews=pre_process_rating(reviews)
      return pred
        # return a list containing all the ratings predicted by the trained model

def weighted_sampler(ratings):
  #weight =[0.2,0.35,0.10,0.2,0.15]
  weight =[0.15,0.35,0.15,0.15,0.2]
  #weight =[0.2,0.35,0.175,0.175,0.1]
  samples_weight = np.array([weight[t] for t in ratings])
  samples_weight = torch.from_numpy(samples_weight)
  samples_weigth = samples_weight.double()
  sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
  return sampler

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs=128,23
    train_data=pd.read_csv(train_file)
    test_data=pd.read_csv(test_file)
    #train_data=over_sample_data(train_data)
    train_reviews=preprocess_data(train_data)
    test_reviews=preprocess_data(test_data,False)
    train_ratings=pre_process_rating(train_data)
    test_ratings=pre_process_rating(test_data)
    sampler=weighted_sampler(train_ratings)

    model=NeuralNet(train_reviews,train_ratings)
    model.build_nn()
    model.train_nn(batch_size,epochs,sampler)
    return model.predict(test_reviews)
main("./train.csv","./gold_test.csv")
