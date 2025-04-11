import streamlit as stm
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import re

nltk.download('punkt_tab')
nltk.download('stopwords')

stemmer = PorterStemmer()

cv = pickle.load(open('cv_vectorizer.pkl', 'rb'))
model = pickle.load(open('model_xgb.pkl', 'rb'))

def text_transform(review):
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = " ".join(review)
    return review

stm.title('Amazon Echo review : Sentiment Analysis')

review = stm.text_area("Enter your review: ")

if stm.button('predict'):
  if review == "":
    stm.header("mail box is empty")
  else:   
    #1. preprocess
    transformed_review = text_transform(review)
    #2. vectorize
    vector_input = cv.transform([transformed_review])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. display
    if result == 1:
      stm.header("Positive Review")
    else:
      stm.header("Negative Review")