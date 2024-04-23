#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Python
# 
# For this problem we will be working with the [Titanic Data Set from Kaggle](https://www.kaggle.com/c/titanic). This is a very famous data set  in machine learning! 
# 
# We'll be trying to predict a classification- survival or deceased.
# Let's begin our understanding of implementing Logistic Regression in Python for classification.
# 
# We'll use a "semi-cleaned" version of the titanic data set, if you use the data set hosted directly on Kaggle, you may need to do some additional cleaning not shown in this lecture notebook.
# 
# ## Import Libraries
# Let's import some libraries to get started!

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## The Data
#  
#  Let's start by reading in the titanic_train.csv file into a pandas dataframe.

# In[2]:


train = pd.read_csv('titanic_train.csv')


# In[3]:


train.head()


# In[4]:


train.columns


# # Exploratory Data Analysis
# 
# Let's begin some exploratory data analysis! We'll start by checking out missing data!
# 
# ## Missing Data
# 
# We can use seaborn to create a simple heatmap to see where we are missing data!

# In[5]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# 
# Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# In[6]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')


# In[7]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# * The plot suggests that many males in the train did not survive
# * However this might be caused by the fact that the train had a lot of males than females

# Let us further check the proportion of males compared to females

# In[8]:


Total_passengers = train['PassengerId'].count()
Total_males = train['PassengerId'][train['Sex']=='male'].count()
Total_females = train['PassengerId'][train['Sex']=='female'].count()
print(f'{Total_passengers} passengers ')
print(f'{Total_males} were males ')
print(f'{Total_females} were females ')


# * Males were approximately twice as much as the number of females

# In[9]:


BySurvival= train.groupby('Survived')['Sex'].value_counts()
BySurvival


# * from $577$ males, only $109$ survived
# * from $314$ males, only $233$ survived

# * let us calculate the survival rate per gender

# * out of $891$ passengers, only $342$ survived

# In[10]:


Total_Survival = (342/891)*100
print(f'Just above {int(Total_Survival)}% of passengers survived "\U0001F622"')


# In[11]:


M_survival_rate = (109/577)*100
F_survival_rate = (233/314)*100

print(f'There is {M_survival_rate:.2f}% chance that a male will survive')
print(f'There is {F_survival_rate:.2f}% chance that a female will survive')


# * According to the survival rate of passengers;
# * females are more likely to survive than males
# * This does not make sence to me, since we know that males are more stronger and I was expecting their majority to survive

# * Apart from Gender, maybe there are other factors affecting the survival of a passenger

# Let us investigate the survival based on the passenger class

# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# * The plot above shows that passengers in the third class died the most 
# * This might suggest that the 3rd class is less safer than other classes, of course it is cheaper
# * However, it might be the case that we had a lot of people in the 3rd class

# let us check the number of people in the 3rd class compared to other classes:

# In[13]:


ByPclass= train.groupby('Pclass')['Survived'].value_counts()
ByPclass


# We further calculate the survival rate based on the passenger class

# In[14]:


class_1 = (136/(136+80))*100
class_2 = (87/(87+97))*100
class_3 = (119/(119+372))*100

print(f'First_class: {class_1:.2f}% chance of surviving')
print(f'Second_class: {class_2:.2f}% chance of surviving')
print(f'Third_class: {class_3:.2f}% chance of surviving')


# * It seems that the 1st class is much safer than any other passenger class

# lets further investigate how the age can affect the survival

# In[15]:


# Age Distribution
sns.displot(train['Age'].dropna(),kde=True,color='darkred',bins=30,alpha=0.7)


# * It looks like many passengers had an age between 20-30 years old

# In[16]:


train['Age'].describe()


# * The average age is 30 years
# * The oldest person is 80 years old
# * The youngest person is less than a year old (5 months old baby)

# Let us check the age distribution based of the passenger class

# The average age per passenger class:

# In[17]:


ByPclass = train.groupby('Pclass')['Age'].mean()
ByPclass


# * We can see the wealthier passengers in the higher classes tend to be older, which makes sense.
# * We see that probably many young people died because there is a lot of them in the 3rd class 

# We further check how does travelling with a companion affected the survival

# This is probably not a variable that is likely to influence the survival of a passenger

# In[18]:


# sibling/spouse
sns.countplot(x='SibSp',data=train)


# * Many passengers were travelling alone

# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# In[19]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# We can see the wealthier passengers in the higher classes tend to be older, which makes sense since older people are more likely to have worked and accumulated a lot of wealth.
# 

# We'll use these average age values to impute based on Pclass for Age.

# In[20]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 29

        else:
            return 25

    else:
        return Age


# Now apply the above function!

# In[21]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[22]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# * We have amputated the age using the mean age per passenger class
# * Now let us remove the Cabin since we have a lot of missing values

# In[23]:


train.drop('Cabin',axis=1,inplace=True)


# In[24]:


train.head()


# In[25]:


train.dropna(inplace=True)


# In[26]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Nothing is missing anymore

# # The Data is now cleaned

# In[27]:


train.head()


# * Now we need to convert categorical Data into data variable(s)

# In[28]:


#Now we need to convert categorical Data into data variable(s)
pd.get_dummies(train['Sex'])


# But we still have a slight problem, the female column is the perfect predictor of the male column and visa-vesa
# This will be a problem to our Machine learning model since these variables are perfectly (linearly) correlated (have the same information) \
# We need to drop one column

# In[29]:


#drop one column
pd.get_dummies(train['Sex'], drop_first=True)


# In[30]:


sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[31]:


sex.head()


# Do the same for the "Embarked" column

# In[32]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[33]:


# Add the new columns to the 'train' DataFrame
train = pd.concat([train, sex, embark], axis=1)


# In[34]:


train.head(2)


# In[35]:


# Drop the feature columns
train.drop(['Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)


# In[36]:


train.head()


# In[37]:


# Drop 'PassengerId', its not useful. its just a number of passenegrs
train.drop('PassengerId', axis=1, inplace=True)


# In[38]:


train.head()


# This looks perfect for the Machine Learning Algorithm (All the Data is numerical)

# Great! Our data is ready for our model!
# 
# # Building a Logistic Regression model 
# # To predict if a passenger Survived Based on the other varibles
# 
# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).
# 

# In[39]:


#split the Data into X and y
X = train.drop('Survived', axis=1)

y = train['Survived']


# ## Train Test Split

# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[42]:


# import linearRegression Model
from sklearn.linear_model import LogisticRegression


# In[43]:


# instantiate
logmodel = LogisticRegression()


# In[44]:


logmodel.fit(X_train, y_train)


# In[45]:


predictions = logmodel.predict(X_test)
predictions


# * Let us compare the predictions with the original Survival:

# In[46]:


Survival = np.array(y_test)
Survival


# * The model's prediction is not bad from what I see above

# We will evaluate the model to check its accuracy

# ## Evaluation

# We can check precision,recall,f1-score using classification report!

# In[47]:


from sklearn.metrics import classification_report


# In[48]:


print(classification_report(y_test, predictions))


# * Indeed the Model has a high precision
# * It is $83$% accurate

# we can also use a Confusion Matrix

# In[49]:


from sklearn.metrics import confusion_matrix


# In[50]:


confusion_matrix(y_test, predictions)


# Not so bad! We might want to explore other feature engineering and the other titanic_text.csv file, some suggestions for feature engineering:
# 
# * Try grabbing the Title (Dr.,Mr.,Mrs,etc..) from the name as a feature
# * Maybe the Cabin letter could be a feature
# * In future we will look at the other info we can get from the ticket?
# 

# In[ ]:





# In[ ]:





# In[ ]:




