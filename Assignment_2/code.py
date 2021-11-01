# ADD THE LIBRARIES YOU'LL NEED
import pandas as pd
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from gensim.models import FastText
import fasttext
import fasttext.util
from tensorflow.keras import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Flatten
from keras.initializers import Constant
from keras.models import model_from_json
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
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
'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''

vocab_embedding=dict()

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

def under_sample_data(train):
    undersample = RandomUnderSampler(random_state=777)
    X_over, y_over = undersample.fit_resample(train[['reviews','ratings']], train['ratings'])
    columnnames=['reviews','ratings']
    df = pd.DataFrame(X_over, columns=columnnames)
    df['ratings'].value_counts()
    train=df
    train = train.sample(frac=1).reset_index(drop=True)
    return train


def pre_process_rating(data):
  ratings = data["ratings"].values.tolist()
  ratings1=list()
  for rating in ratings:
    replace_list=[0,0,0,0,0]
    replace_list[rating-1]=1
    ratings1.append(replace_list)
  ratings=ratings1
  #ratings=[x-1 for x in ratings]
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
    print(elem)
    review=word_tokenize(elem)
    reviews1.append(review)
  reviews=reviews1
  return reviews
    # return the reviews after performing tokenization


def perform_padding(data):
  reviews=data
  review_length=29
  print(review_length)
  reviews= sequence.pad_sequences(reviews, maxlen=review_length,padding="post")
  #print(reviews[2])
  #reviews.shape
  return reviews
    # return the reviews after padding the reviews to maximum length

def preprocess_data(data,is_train=True):
    # make all the following function calls on your data
    # EXAMPLE:->
      
      if(is_train):  
        review = data["reviews"].values.tolist()
      else:
          review=data
      review = convert_to_lower(review)
      review = remove_punctuation(review)
      review = remove_stopwords(review)
      review = perform_tokenization(review)
      review = encode_data(review)
      review = perform_padding(review)
        
      return review



def softmax_activation(x):
  num=K.exp(x-K.reshape(K.max(x,axis=1),(K.shape(x)[0],1)))
  norm=K.reshape(K.sum(num,axis=1),(K.shape(x)[0],1))
  return num/norm
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)



class NeuralNet:

    def __init__(self, reviews, ratings):

        self.reviews = reviews
        self.ratings = ratings



    def build_nn(self):
      with open('./vocab.json') as json_file: 
        vocab_sorted = json.load(json_file)
      vocab_size=len(vocab_sorted)
      #vocab=model
      vocab_embedding = fasttext.load_model('./crawl-300d-2M-subword/crawl-300d-2M-subword.bin')
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
      #print(embedding_matrix[18])
      #print(embedding_matrix[19])
      #print(vocab)

      self.model = Sequential()
      self.model.add(Input(shape=(29,)))
      self.model.add(Embedding(vocab_size+1,embedding_dim,input_length=29,trainable=True,embeddings_initializer=Constant(embedding_matrix)))
      #model.add(Embedding(vocab_size+1,embedding_dim,input_length=1000))
      #model.add(LSTM(100))
      self.model.add(Dropout(0.7))
      self.model.add(Flatten())
      self.model.add(Dense(50,activation='sigmoid'))
      self.model.add(Dropout(0.5))
      #model.add(Dense(70,activation='sigmoid'))
      #model.add(Dropout(0.5))
      #model.add(Dense(25,activation='relu'))
      #model.add(Dropout(0.5))
      self.model.add(Dense(5,activation=softmax_activation))
      self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        #add the input and output layer here; you can use either tensorflow or pytorch

    def train_nn(self,batch_size,epochs):
      reviews_train, reviews_test, ratings_train, ratings_test = train_test_split(self.reviews, self.ratings, train_size=0.75)
      self.model.fit(np.array(reviews_train),np.array(ratings_train),epochs=2,batch_size=256,validation_data=(np.array(reviews_test),np.array(ratings_test)))

        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy

    def predict(self, reviews):
      #test_reviews=pre_process_rating(reviews)
      return np.argmax(self.model.predict(np.array(reviews)),axis=-1)
        # return a list containing all the ratings predicted by the trained model


    
# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs=128,50
    train_data=pd.read_csv(train_file)
    test_data=pd.read_csv(test_file)
    train_data=over_sample_data(train_data)
    train_reviews=preprocess_data(train_data)
    test_reviews=preprocess_data(test_data)
    train_ratings=pre_process_rating(train_data)
    test_ratings=pre_process_rating(test_data)

    model=NeuralNet(train_reviews,train_ratings)
    model.build_nn()
    model.train_nn(batch_size,epochs)

    return model.predict(test_reviews)
#main('train','test')