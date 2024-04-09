#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# These imports provide a strong foundation for data analysis, visualization, and natural language processing tasks in Python.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# # Reading the Dataset

# In[2]:


df=pd.read_csv('twitter_data.csv')


# The Twitter dataset for hate speech detection is a collection of tweets labeled as either containing hate speech or not. It is used as a benchmark dataset for developing and evaluating machine learning models to detect hate speech in text. The dataset typically includes text from tweets along with labels indicating whether each tweet is classified as hate speech or not. Researchers and developers use this dataset to train and test machine learning models to automatically identify hate speech in social media content.

# In[3]:


df


# In[89]:


df.index


# In[4]:


df.head()


# In[5]:


df.tail()


# In[90]:


df.count()


# In[6]:


df.shape


# In[7]:


df.sample(10)


# In[8]:


#Checking The Information.
df.info()


# Their are "1" object columns and "6" numerical(Integer) Columns

# # Checking The Null Values

# In[9]:


df.isnull().sum()


# In[10]:


#Describing The Dataset
df.describe()


# In[11]:


# Describing the data including categorical columns
df.describe(include='object')


# # visualising The Data 

# In[12]:


sns.countplot(x='hate_speech',data=df)
plt.title('Distribution of Hate speech')
plt.xlabel('Plotting of hate speech')
plt.ylabel('Number of Hate speech')
plt.show()


# In[14]:


plt.figure(figsize=(8, 6))
df['offensive_language'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Distribution of offensive_language')
plt.ylabel('Percentage')
plt.show()


# In[95]:


df.dtypes


# In[99]:


plt.figure(figsize=(10,5))
df['hate_speech'].value_counts().plot(kind='barh')
plt.title('Hate Speech Observation',size=20,c='Green',fontweight='bold')
plt.show()


# In[98]:


sns.histplot(df['tweet'], kde=True, color='r')
plt.title('Varities of tweet',size=20,c='Green',fontweight='bold')
plt.xlabel('sentiments')
plt.ylabel('numbers')
plt.xticks(rotation='vertical',color='Black',fontsize=10)
plt.yticks(color='Black',fontsize=10)
plt.grid()
plt.show()


# # Using of NLP

# In[15]:


import nltk
import re
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")


# import nltk: This imports the Natural Language Toolkit (NLTK), a library for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources, such as WordNet.
# 
# import re: This imports the regular expression (regex) module, which is used for pattern matching in strings. It's commonly used for text preprocessing tasks like cleaning and tokenization.
# 
# from nltk.corpus import stopwords: This imports the stopwords corpus from NLTK, which contains a list of common words (e.g., "the," "is," "and") that are often removed from text during natural language processing tasks to focus on more meaningful words.
# 
# stopword=set(stopwords.words('english')): This line initializes a set of English stopwords using NLTK's stopwords corpus.
# 
# stemmer = nltk.SnowballStemmer("english"): This line creates a Snowball stemmer for English, which is used to reduce words to their base or root form (e.g., "running" to "run"). Stemming can help in text normalization and reducing the dimensionality of the feature space in natural language processing tasks.

# In[16]:


from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
SW=stopwords.words('english')


# from nltk.stem import PorterStemmer: This imports the PorterStemmer class from NLTK's stem module. The Porter stemming algorithm is a widely used method for stemming words in English.
# 
# from nltk.corpus import stopwords: This imports the stopwords corpus from NLTK. Stopwords are common words (e.g., "the," "is," "and") that are often removed from text during natural language processing tasks.
# 
# SW=stopwords.words('english'): This line creates a list SW containing English stopwords from the NLTK stopwords corpus. These stopwords can be used to filter out common words from text data.

# In[17]:


SW


# In[18]:


from nltk.corpus import brown
brown.words()


# In[19]:


punctuation


# In[20]:


ps=PorterStemmer()


# In[21]:


df.head()


# In[22]:


df.sample(10)


# In[23]:


def preprocess (msg):
    no_punct = ''.join([char.lower() for char in msg if char not in punctuation])
    tokenized = no_punct.split()
    stem_nostop = [ps.stem(char) for char in tokenized if char not in SW]
    return stem_nostop 


# In[24]:


df['new_tweet']=df['tweet'].apply(lambda x:preprocess(x))


# In[25]:


df.head()


# In[26]:


tfidf_vec=TfidfVectorizer(analyzer=preprocess)


# If preprocess is a custom function you've defined for text preprocessing, you can use it as a tokenizer within the TfidfVectorizer by passing it to the tokenizer parameter. 

# In[27]:


def length(msg):
    total_len=len(msg)-msg.count(' ')
    return total_len


# In[28]:


df['tweet']=df['tweet'].apply(lambda x:length(x))


# In[29]:


df


# In[30]:


#checking the unique value
df['count'].unique()


# In[31]:


print("The percentage of data that is null:")
df.isnull().sum()/len(df)*100


# # KDE Plotting

# # scale down the value so that value get less and Distribution will be same

# In[33]:


df['class'].plot(kind='kde')


# In[34]:


df['offensive_language'].plot(kind='kde')


# In[35]:


df.skew()


# # using of counter Vector

# CountVectorizer object, which is used to convert a collection of text documents into a matrix of token counts. Each row of the matrix represents a document in the collection, and each column represents a unique token in the corpus, with the matrix value indicating the count of the token in the corresponding document.

# In[36]:


X=df.drop('Unnamed: 0',axis=1)


# In[37]:


X


# # Using of power Transformer

# By transforming the data, you can often improve the performance and stability of your machine learning models, especially those sensitive to the distribution or scale of the input features.

# In[38]:


from sklearn.preprocessing import PowerTransformer


# In[39]:


pt=PowerTransformer(method='yeo-johnson')


# In[40]:


df[['count','hate_speech','class']]=pt.fit_transform(df[['count','hate_speech','class']])


# In[41]:


df.head()


# # Training ,Testing and spliting of Data

# In[47]:


x=df['class']


# In[48]:


x


# In[44]:


y=df['count']


# In[45]:


y


# In[59]:


x_test.head()


# In[61]:


y_test


# # Accuracy of Model

# In[67]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your Twitter data (assuming a CSV file with columns for tweet text and label)
data = pd.read_csv("twitter_data.csv")

# Separate features (text) and target variable (label)
text = data["tweet"]
label = data["count"]  # Assuming "label" is the column containing hate speech classification (0/1)

# Preprocess text (e.g., remove stop words, lowercase)
# ... (Implement your text preprocessing steps here)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(text)

# Split data into training and testing sets (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Create the Decision Tree Classifier model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate model performance (accuracy in this example)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[70]:


from sklearn.metrics import r2_score,mean_squared_error


# R2 score predict the model prediction by input variable and accuracy should be less than 1

# In[72]:


r2_score(y_test,y_pred)*100


# # Error Check

# In[73]:


mean_squared_error(y_test,y_pred)


# It have good Accuracy .

# # Why Pairplot?

# A pairplot creates a matrix of scatter plots, where each subplot shows the relationship between two variables. The diagonal plots show the distribution of each variable using histograms.

# # Advantages
Comprehensive View: Pairplots provide a compact overview of relationships between many variables in a single plot, avoiding the need for numerous individual scatter plots.
Identifying Patterns: The visual nature of pairplots makes it easier to spot patterns and trends that might be less obvious in tables or correlation matrices.
# In[91]:


sns.pairplot(data=df)

plt.show()


# In[92]:


sns.pairplot(data=df,hue='hate_speech')

plt.show()


# In[93]:


df.corr()


# # why Heatmap?

#  Pay attention to the colormap used. It typically ranges from a low value (often blue) to a high value (often red), with a neutral color (like white or gray) in the middle for some colormaps. Understanding this gradient is crucial for interpreting the intensity of the colors.

# In[94]:


sns.heatmap(df.corr(),annot=True)


# # Conclusion: Unveiling Hate Speech on Twitter with NLP
# This project investigated the effectiveness of Natural Language Processing (NLP) techniques in identifying hate speech on the Twitter platform. We employed a decision tree classification model, trained on Twitter data with TF-IDF features, to achieve an accuracy of [88.6%] in classifying hate speech. This success demonstrates the potential of NLP for creating a safer Twitter environment.

# Overall, this project highlights the promise of NLP for tackling hate speech on Twitter. By continuously improving NLP techniques and addressing ethical concerns, we can create a safer and more inclusive Twitter experience for all users.

# # Future scope

# Integrating the NLP model with Twitter's content moderation systems could enable real-time detection and flagging of hate speech, fostering a more respectful online environment.
# Domain-Specific Techniques: Exploring NLP advancements focused on social media language, including sentiment analysis tailored to Twitter's unique dynamics, can enhance detection accuracy.
# Adapting to Evolving Language: Hate speech often utilizes novel slang, memes, or coded language. Continuous adaptation of the NLP model through regular retraining with fresh Twitter data is crucial.
