
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

#  load data into veriable
data_train = pd.read_csv("C:/Users/Abhishek/Desktop/ds_data_big/ds_data/data_train.csv")
data_train.head()


# In[4]:


#  fix what are you predicting and what feature you are using

column_target = ['target']
column_train = ['id', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'num11', 'num12',
                'num13', 'num14','num15', 'num16', 'num17', 'num18', 'num19', 'num20', 'num21', 'num22', 'num23','der1',
                'der2', 'der3', 'der4', 'der5', 'der6', 'der7', 'der8', 'der9', 'der10', 'der11', 'der12', 'der13',
                'der14', 'der15', 'der16', 'der17', 'der18', 'der19', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6',
                'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14']


# In[5]:


x = data_train[column_train]
y = data_train[column_target]


# In[6]:


#  check whether dataset contains null value
x.isnull().sum()


# In[7]:


# replacing nan value in dataset wirh mean and categoric value
x['num18'] = x['num18'].fillna(x['num18'].mean())
x['num19'] = x['num18'].fillna(x['num19'].mean())
x['num20'] = x['num20'].fillna(x['num20'].mean())
x['num22'] = x['num22'].fillna(x['num22'].mean())
x['cat1'].fillna(value=4.0, inplace=True)
x['cat2'].fillna(value=1.0, inplace=True)
x['cat3'].fillna(value=5.0, inplace=True)
x['cat4'].fillna(value=1.0, inplace=True)
x['cat5'].fillna(value=0.0, inplace=True)
x['cat6'].fillna(value=0.0, inplace=True)
x['cat8'].fillna(value=0.0, inplace=True)
x['cat10'].fillna(value=0.0, inplace=True)
x['cat12'].fillna(value=4.0, inplace=True)


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


# In[9]:


# spliting test and train data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=67)


# In[12]:


#  using logistic regression which seperate data with line
clf = LogisticRegression()
model = clf.fit(x_train, y_train)


# In[13]:


#  making prediction using model
pre = model.predict(x_test)

org = np.array(y_test).flatten()


# In[14]:


score = accuracy_score(pre, org)*100
score


# In[15]:


# importing test data
data_test = pd.read_csv("C:/Users/Abhishek/Desktop/ds_data_big/ds_data/data_test.csv")


# In[16]:


column_test = ['id', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'num11', 'num12',
                'num13', 'num14','num15', 'num16', 'num17', 'num18', 'num19', 'num20', 'num21', 'num22', 'num23','der1',
                'der2', 'der3', 'der4', 'der5', 'der6', 'der7', 'der8', 'der9', 'der10', 'der11', 'der12', 'der13',
                'der14', 'der15', 'der16', 'der17', 'der18', 'der19', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6',
                'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14']


# In[17]:


test = data_test[column_test]


# In[18]:


test.isnull().sum()


# In[19]:


test['num18'] = test['num18'].fillna(test['num18'].mean())
test['num22'] = test['num22'].fillna(test['num22'].mean())
test['cat1'].fillna(value=4.0, inplace=True)
test['cat2'].fillna(value=1.0, inplace=True)
test['cat3'].fillna(value=5.0, inplace=True)
test['cat4'].fillna(value=1.0, inplace=True)
test['cat5'].fillna(value=0.0, inplace=True)
test['cat6'].fillna(value=0.0, inplace=True)
test['cat8'].fillna(value=0.0, inplace=True)
test['cat10'].fillna(value=0.0, inplace=True)
test['cat12'].fillna(value=4.0, inplace=True)


# In[22]:


pre_test = model.predict(test[1:1000])
pre_test

