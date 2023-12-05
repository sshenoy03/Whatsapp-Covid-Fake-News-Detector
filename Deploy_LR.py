import pandas as pd
import streamlit as st 
import pickle 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import  LogisticRegression
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline


#Cleaning input data 
stops = set(stopwords.words("english"))
def cleantext(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+",' ',text)    
    text = re.sub(r"www(\S)+",' ',text)
    text = re.sub(r"&",' and ',text)  
    tx = text.replace('&amp',' ')
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text)
    text = text.split()
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text



# load the model from disk

st.title('Whatsapp COVID-19 Fake News Classification app ')

sentence = st.text_area("Enter your COVID relates  news content message  here", "Some news",height=200)
predict_btt = st.button("predict")

if predict_btt:
    clean_text = []
    filename="C:\\Users\\sah-1\\SML PROJECT\\best_LR_model.sav"
    model = pickle.load(open(filename, 'rb'))
    text=[sentence]
    text=[cleantext(str) for str in text]
    predtext=model.predict(text)
    
    prediction_class=predtext[0]
    
    if prediction_class == 0:
        st.success('This is not a fake news')
    if prediction_class == 1:
        st.warning('This is a fake news')

        
