
# coding: utf-8

# In[79]:


import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.externals import joblib


# In[6]:


nltk.download('stopwords')
nltk.download('punkt')


# In[75]:


from sklearn.externals import joblib
loaded_model = joblib.load('LRmodel.pkl')
vectorizer = joblib.load('vectorizer.pkl')
labelEncoder = joblib.load('labelEncoder.pkl')


# In[80]:


#Data cleaning
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your chose
def clean_text(text, ):

    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

    def remove_special_characters(text, characters=string.punctuation.replace('-', '')):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))

    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])

    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)

    text = re.sub(r'[0-9]+', '', text)
    text = text.lower() #lowercase
    text = stem_text(text) #stemming
    text = remove_special_characters(text) #remove punctuation and symbols
    text = remove_stopwords(text) #remove stopwords
    #text.strip(' ') # strip white spaces again?

    return text


# In[102]:


def predictFromFile(filename):
    data = pd.read_csv(filename)
    data.columns = ['Product','Description']
    
    features = pd.DataFrame()
    features['Description'] = file.Description.apply(clean_text)
    
    
    #vectorize using the save count vectorizer
    features = vectorizer.transform(features.Description)
    
    #make prediction using loaded model
    y_pred = loaded_model.predict(features)
    
    #inverse transform to get the category label
    y_pred = labelEncoder.inverse_transform(y_pred)
    file['Predicted Category'] = y_pred
    #save the file as csv
    file.to_csv('Output_of_model')
    print('File Generated!')
    return file


# In[103]:


data = predictFromFile('test_table.csv')


# In[104]:


def predictFromText(text):
    text = pd.Series(text)
    text = text.apply(clean_text)
    text = vectorizer.transform(text)
    y_pred = loaded_model.predict(text)
    y_pred = labelEncoder.inverse_transform(y_pred)
    return y_pred[0]


# In[107]:


predictFromText('Radial Desk White A2010 1600 x 1600mm')


# In[106]:


file.tail()

