#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


titanic_data = pd.read_csv('train.csv')


# In[3]:


titanic_data.describe()


# In[4]:


import seaborn as sns
sns.heatmap(titanic_data.corr(), cmap="YlGnBu")
plt.show()


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2)
for train_indices, test_indices in split.split(titanic_data,titanic_data[["Survived","Pclass","Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[6]:


strat_test_set


# In[7]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.show()


# In[8]:


strat_test_set.info()


# In[9]:


women = titanic_data.loc[titanic_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[10]:


men = titanic_data.loc[titanic_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# In[11]:


titanic_data.shape


# In[12]:


titanic_data .isnull().sum()


# In[13]:


titanic_data  = titanic_data.drop (columns='Cabin', axis=1)


# In[14]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[15]:


print(titanic_data['Embarked'].mode())


# In[16]:


print(titanic_data['Embarked'].mode()[0])


# In[17]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[18]:


titanic_data .isnull().sum()


# In[19]:


titanic_data['Survived'].value_counts()


# In[20]:


sns.countplot(x='Survived', data = titanic_data)


# In[21]:


sns.countplot(x='Sex',hue='Survived',data = titanic_data)


# In[22]:


sns.countplot(x='Pclass', data=titanic_data)


# In[23]:


sns.countplot(x='Pclass',hue='Survived', data=titanic_data)


# In[24]:


titanic_data['Embarked'].value_counts()


# In[25]:


titanic_data['Sex'].value_counts()


# In[26]:


p=[577,314]
mylabels = titanic_data.index
sns.countplot(x='Survived',data=titanic_data,width=0.4)
plt.title('No of People Survived by Gender')
#plt.pie(p,labels=mylabels)
plt.legend(['dead'],loc='upper right')
plt.show()


# In[27]:


titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[28]:


titanic_data.head()


# In[29]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis=1)
Y = titanic_data['Survived']


# In[30]:


print(X)


# In[31]:


print(Y)


# In[32]:


X_train, X_test, Y_train,Y_test, = train_test_split(X,Y, test_size=0.2,random_state=2)


# In[33]:


print(X.shape, X_train.shape, X_test.shape)


# In[34]:


sns.pairplot(data=titanic_data, hue='Survived')


# In[35]:


model = LogisticRegression()


# In[36]:


model.fit(X_train, Y_train)


# In[37]:


X_train_prediction = model.predict(X_train)


# In[38]:


print(X_train_prediction)


# In[39]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data:', training_data_accuracy)


# In[40]:


X_test_prediction = model.predict(X_test)


# In[41]:


print(X_test_prediction)


# In[42]:


test_data_accuracy = accuracy_score(Y_test ,X_test_prediction)
print('Accuracy score of test data:', test_data_accuracy)


# In[ ]:


#Hence our Model Predicted The Survival Rate of a Person with "78%" Accuracy

