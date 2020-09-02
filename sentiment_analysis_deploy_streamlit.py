import streamlit as st
st.title("Machine Learning Model")
st.subheader("This is a Review Classifier. Try it Out!")

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

# Wordcloud
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

comment_words = '' 
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df['Text']:
  val = str(val) 
  tokens = val.split() 

  for i in range(len(tokens)): 
    tokens[i] = tokens[i].lower()     
  comment_words += " ".join(tokens)+" "

alice_coloring = np.array(Image.open("img.png"))

wc = WordCloud( background_color="white",mask=alice_coloring,width = 600, height = 600,
               stopwords=stopwords, max_font_size=30)
# generate word cloud
wc.generate(comment_words)

# create coloring from image
image_colors = ImageColorGenerator(alice_coloring)
print(alice_coloring.shape)
# show
fig, axes = plt.subplots(figsize=(6,6))
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor

axes.axis("off") 
plt.tight_layout(pad = 0) 
#axes.imshow(wc, interpolation="bilinear")
axes.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
#axes.imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")

if(user_input != ""):
 st.pyplot()

y_pred = text_model.predict([user_input])
if(y_pred == 2):
  st.write('Positive Review')
elif (y_pred == 1):
  st.write('Negative Review')
else:
  st.write('Negative Review')
