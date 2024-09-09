#!/usr/bin/env python
# coding: utf-8

# In[141]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[142]:


import nltk
nltk.download('stopwords')


# In[143]:


print(stopwords.words('english'))


# In[144]:


traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')


# In[145]:


traindf


# In[146]:


print(f'shape of train dataset',traindf.shape)
print(f'shape of test dataset',testdf.shape)


# In[147]:


traindf.isnull().sum()
testdf.isnull().sum()


# In[148]:


traindf = traindf.fillna('')
testdf = testdf.fillna('')


# In[149]:


traindf.isnull().sum()


# In[150]:


#merging the authors name and the title
traindf['content'] = traindf['title']+traindf['author']

traindf['content']


# In[151]:


#seperating the data and label
X = traindf.drop('label',axis =1)
Y = traindf.label



# In[152]:


print(X)
print(Y)


# In[153]:


#stemming is process to reduce a word to its root
port_stem = PorterStemmer()


# In[154]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[155]:


traindf['content'] = traindf['content'].apply(stemming)


# In[156]:


print(traindf['content'])


# In[157]:


X = traindf['content'].values
Y = traindf['label'].values


# In[158]:


print(X)


# In[159]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)


# In[160]:


print(X)


# In[161]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2 , stratify=Y,random_state=2)


# In[162]:


model = LogisticRegression()


# In[163]:


model.fit(x_train,y_train)


# In[164]:


pred = model.predict(x_train)
acc = accuracy_score(pred,y_train)


# In[165]:


print(acc)


# In[166]:


pred1 = model.predict(x_test)
acc1 = accuracy_score(pred1,y_test)


# In[167]:


print(acc1)


# In[168]:


x_new = x_test[0]
prediction =model.predict(x_new)
print(prediction)

if(prediction[0]==0):
    print("The news is real")
else:
    print("The news is fake")


# In[170]:


print(y_test[0])


# In[ ]:




