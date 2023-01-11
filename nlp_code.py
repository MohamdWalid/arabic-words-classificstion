

from google.colab import drive
import cv2
drive.mount('/content/drive/')

#!unzip -q "/content/drive/MyDrive/model_a3/ar_wiki_word2vec-300.rar"
#!unrar x -Y "/content/drive/MyDrive/Google/ar_wiki_word2vec-300.rar" "/content/drive/MyDrive/model_a3/"

#!sudo cp -v -r  "/content/drive/MyDrive/model_a3/ar_wiki_word2vec-300.rar"  "/content/drive/MyDrive/model_a3"

"""#**NLTK install & Library**"""

!pip install nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = list(stopwords.words('arabic'))
length_stop_words = len(stop_words)
print(stop_words)

"""#**Preprocessing & LOAD DATA**"""

from nltk.sem.logic import ExpectedMoreTokensException
import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/a3_data/all.test.features.csv')

#Drop NAN
df.dropna(inplace=True)

#Remove Punkt
punkt = ['+','-','.','*','/','"\"','=','-','_',')','(','&','^','%','$','#','@','!','~','[',']','{','}','|','"',':',';',',','<','>','.','/','?','\n','>=','<=','==']
for i in range(0,len(punkt)):
   df.drop((df.loc[df.Column1 ==punkt[i],['Column1' ,'Column37']].index),axis=0,inplace=True)

#Remove Stop_Words
for i in range(0,length_stop_words):
   df.drop((df.loc[df.Column1 == stop_words[i] ,['Column1' ,'Column37']].index),axis=0,inplace=True)

data = np.array(df['Column1'])
label = np.array(df['Column37'])

print(f"all data: {set(data)}")

print(f"all Labels: {set(label)}")

print(len(data))
print(len(label))

"""#**Load Arabic pretrained word2vec model.**"""

import gensim
from gensim.models import Word2Vec , KeyedVectors
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
import gensim
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
mod = gensim.models.Word2Vec.load('/content/drive/MyDrive/model_a3/ar_wiki_word2vec')

"""#**Remove English words and numbers from Data**"""

Words = []
Output = []
for i in range(0,len(data)):
  try:
    mod[data[i]]
    Words.append(data[i])
    Output.append(label[i])
  except:
    pass

print(len(Words))
print(len(Output))

print(f"all data: {set(Words)}")

print(f"all Labels: {set(Output)}")

"""#**Convert Labels From String to Integer Number**"""

Output = list(map(lambda x : 0 if x=='O' else x,Output))
Output = list(map(lambda x : 1 if x=='I-PER' else x,Output))
Output = list(map(lambda x : 2 if x=='B-LOC' else x,Output))
Output = list(map(lambda x : 3 if x=='I-MIS' else x,Output))
Output = list(map(lambda x : 4 if x=='B-MIS' else x,Output))
Output = list(map(lambda x : 5 if x=='I-LOC' else x,Output))
Output = list(map(lambda x : 6 if x=='B-ORG' else x,Output))
Output = list(map(lambda x : 7 if x=='B-PER' else x,Output))
Output = list(map(lambda x : 8 if x=='I-ORG' else x,Output))

print(Output)

"""#**Get Embedding Vector for All Words**
#**Split Data to train and test**
"""

train = []
test = []
label_train = []
label_test= []
for i in range(0,len(Words)):
  if i%5==0:
    test.append(mod[Words[i]])
    label_test.append(Output[i])

  else:
    train.append(mod[Words[i]])
    label_train.append(Output[i])

len(train)

print(len(label_train))

len(test)

print(len(label_test))

label_train=np.array(label_train)
label_test=np.array(label_test)
train=np.array(train)
test=np.array(test)

print(train.shape)
print(label_train.shape)

"""#**Library for LSTM Model**"""

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense

"""#**LSTM Model**"""

model1 = Sequential()
model1.add(LSTM(units=50, return_sequences=True, input_shape=(train.shape[1],1)))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50))
model1.add(Dropout(0.2))
model1.add(Dense(units = 9))
#model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model1.compile(optimizer='adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model1.fit(train, label_train, epochs = 50, batch_size = 32)

"""#**Save LSTM Model**"""

model1.save('/content/drive/MyDrive/model_a3/LSTM_MODEL.h5')

"""#**Test Accuracy**"""

label = list(map(lambda x : np.argmax(x),y_pred))
label = np.array(label)
print(f'accuracy: {np.round((model1.evaluate(test,label_test))[1],2)}')

"""#**First Process For Test Sentence (tokenizer ,stopwords , remove numbers and english words)**"""

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('arabic'))
def process_of_test_sentence(word):
  output = re.sub(r'\s*[A-Za-z1-9]+\b', '.' , word)
  #doc= re.sub(r'[^\w\s]', ' ', doc)
  output= nltk.word_tokenize(output)
  out = [w for w in output if not w in stop_words]
  return out

"""#**Second Process For Test Sentence (Get Label for All Words in Sentence )**"""

def final_output(list1) :
 op_sent=[]
 op=['O','I-PER','B-LOC','I-MIS','B-MIS','I-LOC','B-ORG','B-PER','I-ORG']
 b=mod[list1]
 z= model1.predict(b)
 for q in range(len(z)):
  max = np.argmax(z[q])
  op_sent.append(op[max])
 return(op_sent)

"""#**Third Process For Test Sentence (Print  All Words in Sentence And it is Label)**"""

def words_and_label(pross_sent1,pross_sent2):
     print(f' the word :  it is label')
     for i in range(0,len(pross_sent1)):
      print(f' {pross_sent1[i]} :  {pross_sent2[i]}')

"""#**Test Sentence**"""

sent_test2='إن نظام سولاريس بني اعتمادا على عدة أنظمة تشغيل متفرعة من انظمة تشغيل'
pross_sent2=process_of_test_sentence(sent_test2)
pross_sent3=final_output(pross_sent2)
final_pross= words_and_label(pross_sent2,pross_sent3)