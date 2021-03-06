import streamlit as st
st.title("Machine Learning Model")
st.subheader("This is a Review Classifier. Try it Out!")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="style.css" rel="stylesheet">', unsafe_allow_html=True)    

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

import pandas as pd
import numpy as np

## Import Dataset

df = pd.read_csv('Amazon Review.csv')
df.head()

## Setting Input and Output
df['Text'] = df['Text'].fillna(' ')
X = df.iloc[:,1].values

y = df.iloc[:,0].values


## Preprocessing for Improvement

df["sentiment"] = 1

df.loc[df['Rating']<=2, 'sentiment'] = 0
df.loc[df['Rating']==5, 'sentiment'] = 2

y = df.iloc[:,2].values

## Splitting data into Train and Test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

## Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

text_model = Pipeline([('tfidf',TfidfVectorizer(binary = False,max_df=0.611111111111111,norm = 'l2')),('model',MultinomialNB(alpha = 0.058,fit_prior=True))])

text_model.fit(X_train,y_train)

counts = np.bincount(y_train)
v = np.argmax(counts)
st.subheader("Your Review :  ")

user_input = st.text_area("\n", "")
print()

y_pred = text_model.predict([user_input])
if(y_pred == 2):
  st.write('Positive Review')
elif (y_pred == 1):
  st.write('Negative Review')
else:
  st.write('Negative Review')
print()
