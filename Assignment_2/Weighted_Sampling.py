# ADD THE LIBRARIES YOU'LL NEED
import nltk
nltk.download('stopwords')
from nltk import word_tokenize as tokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import string

import torch

import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import seaborn as sns

torch.manual_seed(0)

'''
About the task:

You are provided with a codeflow- which consists of functions to be implemented(MANDATORY).

You need to implement each of the functions mentioned below, you may add your own function parameters if needed(not to main).
Execute your code using the provided auto.py script(NO EDITS PERMITTED) as your code will be evaluated using an auto-grader.
'''

contractions = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

def remove_contracted_form(text):
    reviews=[]
    for review in text:
        for key in contractions:
            value = contractions[key]
            review = review.replace(key,value)
        reviews.append(review)
    return reviews


def encode_data(text):
    # This function will be used to encode the reviews using a dictionary(created using corpus vocabulary) 
    
    # Example of encoding :"The food was fabulous but pricey" has a vocabulary of 4 words, each one has to be mapped to an integer like: 
    # {'The':1,'food':2,'was':3 'fabulous':4 'but':5 'pricey':6} this vocabulary has to be created for the entire corpus and then be used to 
    # encode the words into integers 

    # return encoded examples
    
    embeddings_dict = {}
    with open("glove.twitter.27B.200d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    
    
    # replacing each word by corresponding vector (only if word found)
    reviews = []
    cur=1
    for review in text:
        vector = []
        for word in review:
            if word in embeddings_dict.keys():
                vector.append(embeddings_dict[word])
            
        reviews.append(vector)
    
    return reviews

def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return [review.lower() for review in text]


def remove_punctuation(text):
    # return the reviews after removing punctuations
    return [''.join([char for char in review if char not in string.punctuation]) for review in text]
    

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    stop_words = set(stopwords.words('english'))
    return [ [word for word in review if word not in stop_words] for review in text ]

def perform_tokenization(text):
    # return the reviews after performing tokenization
    return [ tokenizer(review) for review in text]

max_len=-1

def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    global max_len
    # set max_len only during training
    if max_len==-1:
    	max_len = max([len(review) for review in data])
    	
    print("max length is " , max_len)
    
    const = np.zeros(200)
    
    # padding each sentence by 1D vector of zeros of size 200 to match max len = 29
    for review in data:
        for i in range(max_len-len(review)):
            review.append(const)
        
    return data
    

def preprocess_data(data):

    global max_len

    reviews = data["reviews"]
    
    reviews = convert_to_lower(reviews)
    
    reviews = remove_contracted_form(reviews)
   
    reviews = remove_punctuation(reviews)
    
    reviews = perform_tokenization(reviews)
    
    #reviews = remove_stopwords(reviews)

    reviews = encode_data(reviews)
    
    reviews = perform_padding(reviews)
   
    return torch.Tensor(reviews).reshape(-1,max_len*200)



def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    
    # subtracting max values (for each row) so that it will not give nan
    m,_ = torch.max(x,dim=1,keepdim=True)
    x=x-m
    
    # calculating exponential 
    x = torch.exp(x)

    # row wise summation 
    s = x.sum(dim=1,keepdim=True)
    
    return x/s
    

class NeuralNet:

    def __init__(self, data):

        self.data = data



    def build_nn(self):
        #add the input and output layer here; you can use either tensorflow or pytorch
        global max_len
    	
        model = torch.nn.Sequential(
            torch.nn.Linear(max_len*200,5)
           
        )
        self.model = model
        
    def get_accuracy(self,y_pred,target):
    
        _, y_pred = torch.max(y_pred, dim = 1)
        correct_pred = (y_pred == target).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = acc * 100
        return acc
        
    def get_class_weights(self,train_df):
    
        weights = []

        for i in range(0,5):
            
            count = len([row for row in train_df if row[-1]==i])
            weights.append(1./count)
            
        return torch.tensor(weights)

    def get_data_loader(self,data,batch_size):

        class_weights = self.get_class_weights(data)
        
        target_list = data[:,-1:].type(torch.LongTensor)

        sample_weights = [class_weights[target] for target in target_list]
        sample_weights = torch.tensor(sample_weights)
        sample_weights = sample_weights.double()
        
        weighted_sampler = WeightedRandomSampler(weights=sample_weights,
                                  num_samples=len(sample_weights),replacement=True)

        data_loader = DataLoader(dataset=data, shuffle=False, 
                                  batch_size=batch_size, sampler=weighted_sampler)
        
        return data_loader

    	

    def train_nn(self,batch_size,epochs):
        # write the training loop here; you can use either tensorflow or pytorch
        # print validation accuracy
        
        # last 10% used as validation set
        train_data = torch.cat( (self.data[:30000], self.data[40000:]) )
        val_data = self.data[30000:40000]

        train_loader = self.get_data_loader(train_data,batch_size)
        validation_loader = self.get_data_loader(val_data,batch_size)
        
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        
        # these lists will store loss and accuracy for each epoch , can plot graphs later
        train_loss , train_acc , val_loss , val_acc = [],[],[],[]
        train_predictions_all , val_predictions_all = [],[]
        
        for epoch in range(epochs):
        
            train_epoch_loss=0
            val_epoch_loss=0
            train_epoch_accuracy=0
            val_epoch_accuracy=0
            
            train_pred , val_pred = [],[]
            
            loss_fn  = torch.nn.NLLLoss()
            
            num_itr_train = 0
            
            for i , batch in enumerate(train_loader):
            
            	pred_ratings = self.model(batch[:,:-1])
            	pred_ratings = softmax_activation(pred_ratings)
            	
            	_, y_pred = torch.max(pred_ratings, dim = 1)
            	train_pred.append(y_pred)
            	
            	y_true = batch[:,-1:].reshape(-1,).type(torch.LongTensor)
            	
            	acc = self.get_accuracy(pred_ratings, y_true)
            	
            	pred_ratings = torch.log(pred_ratings)
            	
            	loss = loss_fn(pred_ratings, y_true)
            	
            	train_epoch_loss+=loss
            	
            	num_itr_train += 1
            	
            	train_epoch_accuracy+=acc
            	
            	# zero out all gradients
            	optimizer.zero_grad()
            	
            	#this will calculate gradients for all paramenters
            	loss.backward()
            	
            	optimizer.step()
            
                
               
                
            # no_grad() used since validation will not update model parameters        	
            with torch.no_grad():
            
            
            	num_itr_val = 0
            	
            	for idx , batch in enumerate(validation_loader):

                    
                    pred_ratings = self.model(batch[:,:-1])
                
                    pred_ratings = softmax_activation(pred_ratings)
                    
                    _, y_pred = torch.max(pred_ratings, dim = 1)
                    val_pred.append(y_pred)
                    
                    val_true = batch[:,-1:].reshape(-1,).type(torch.LongTensor)

                    acc=self.get_accuracy(pred_ratings, val_true)
                    
                    pred_ratings = torch.log(pred_ratings)
                
                    
                    loss = loss_fn(pred_ratings , val_true)
                
                    val_epoch_loss+=loss
                    
                    val_epoch_accuracy+=acc
                
                    
                    num_itr_val+=1
            
            
                
      
            train_loss.append(train_epoch_loss/num_itr_train)
            val_loss.append(val_epoch_loss/num_itr_val)
            train_acc.append(train_epoch_accuracy/num_itr_train)
            val_acc.append(train_epoch_accuracy/num_itr_val)
            
            #train_predictions_all.append(torch.cat(train_pred))
            #val_predictions_all.append(torch.cat(val_pred))
            
            
            print("train loss is {} , train acc is {} ,val loss is {} , val acc is {}".format(train_epoch_loss/num_itr_train,train_epoch_accuracy/num_itr_train,val_epoch_loss/num_itr_val,val_epoch_accuracy/num_itr_val))
            
            
        

    def predict(self, reviews):
        # return a list containing all the ratings predicted by the trained model
        pred_ratings = self.model(reviews)
        pred_ratings = softmax_activation(pred_ratings)
        _, test_ratings = torch.max(pred_ratings,dim=1,keepdim=True)
        
        return test_ratings


def do_evaluations(model , train_reviews , train_ratings , test_reviews , test_ratings):

    train_pred = model.predict(train_reviews)
    test_pred = model.predict(test_reviews)
    
    train_pred = [rating+1 for rating in train_pred]
    train_ratings = [rating+1 for rating in train_ratings]
    test_pred = [rating+1 for rating in test_pred]
    test_ratings = [rating+1 for rating in test_ratings]
    
    cf_report_train = classification_report(train_ratings,train_pred , labels=[1,2,3,4,5])
    cf_report_test = classification_report(test_ratings,test_pred, labels=[1,2,3,4,5])
    
    cm_train = confusion_matrix(train_ratings,train_pred)
    sns_plot_train = sns.heatmap(cm_train/np.sum(cm_train), annot=True,cmap="OrRd",fmt="0.2%")
    
    cm_test = confusion_matrix(test_ratings,test_pred)
    sns_plot_test = sns.heatmap(cm_test/np.sum(cm_test), annot=True,cmap="OrRd",fmt="0.2%")
    
    fig_train = sns_plot_train.get_figure()
    fig_test = sns_plot_test.get_figure()
    
    #fig_train.savefig('train_confusion_matrix.png')
    #fig_test.savefig('test_confusion_matrix.png')
    
    file_train = open('train_classification_report.txt','w')
    print(cf_report_train,file=file_train)
    file_train.close()
    
    file_test = open('test_classification_report.txt','w')
    print(cf_report_test,file=file_test)
    file_test.close()
    
    

# DO NOT MODIFY MAIN FUNCTION'S PARAMETERS
def main(train_file, test_file):
    
    batch_size,epochs= 32,15
    

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    
    train_reviews=preprocess_data(train_data)
    
    train_ratings=[]
    for rating in train_data['ratings']:
    	train_ratings.append(rating-1)
    	
    test_reviews=preprocess_data(test_data)
    
    train_ratings = torch.LongTensor(train_ratings).reshape(-1,1)
    train_data = torch.cat((train_reviews,train_ratings),1)

    df_test = pd.read_csv('gold_test.csv')
    test_ratings=[]
    for rating in df_test['ratings']:
    	test_ratings.append(rating-1)
    test_ratings = torch.tensor(test_ratings).reshape(-1,1)

    model=NeuralNet(train_data)
    model.build_nn()
    model.train_nn(batch_size,epochs)
    
    do_evaluations(model , train_reviews , train_ratings , test_reviews , test_ratings)
    
    test_pred = model.predict(test_reviews)
    test_pred = [rating+1 for rating in test_pred]
    
    return test_pred
    
    
