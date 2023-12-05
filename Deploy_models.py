import pandas as pd
import streamlit as st 
import pickle 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import  LogisticRegression
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from keras import backend as K
from nltk.stem.porter import *
from tensorflow.keras.preprocessing.text import one_hot

#Cleaning input data 
stops = set(stopwords.words("english"))
stop=stops
#Performing stemming on the review dataframe
ps = PorterStemmer()

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

def cleaninput(string):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+",' ',text)    
    text = re.sub(r"www(\S)+",' ',text)
    text = re.sub(r"&",' and ',text)  
    tx = text.replace('&amp',' ')
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text)
    text = text.split()
    text = [ps.stem(w) for w in text if not w in stop]
    text = " ".join(text)
    return text

with open("C:\\Users\\sah-1\\SML PROJECT\\one_hot.pickle", 'rb') as handle:
    one_hot= pickle.load(handle)       #vocsize=10000,maxlen=400

LSTM=load_model("C:\\Users\\sah-1\\SML PROJECT\\biLSTM.h5")
filename="C:\\Users\\sah-1\\SML PROJECT\\best_LR_model.sav"
LR = pickle.load(open(filename, 'rb'))
filename="C:\\Users\\sah-1\\SML PROJECT\\SVM_model.sav"
SVM=pickle.load(open(filename, 'rb'))
filename="C:\\Users\\sah-1\\SML PROJECT\\NB_model.sav"
NB=pickle.load(open(filename, 'rb'))

# load the model from disk
# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )

# add_bg_from_local("C:\\Users\\sah-1\\OneDrive\\Pictures\\SML PROJECT\\bg4.jpg")



st.title('Whatsapp COVID-19 Fake News Classification app ')
st.subheader("Curbing Misinformation,One Message at a time ")

sentence = st.text_area("Enter your COVID related news content message  here", "Some news",height=200)
predict_btt = st.button("predict")

if predict_btt:
    clean_text = []
    #LSTM model
    
    text=[cleaninput(sentence)]
    onehot_text=[one_hot(words,10000)for words in text]
    embedded_text=pad_sequences(onehot_text,padding='pre',maxlen=400)
    predtextLSTM=(LSTM.predict(embedded_text) > 0.5).astype("int32")
    prediction_classLSTM=predtextLSTM[0][0]

    #LR model
    
    text=[sentence]
    textml=[cleantext(str) for str in text]
    predtextLR=LR.predict(textml)

    
    prediction_classLR=predtextLR[0]
    
    #SVM model
    
    predtextSVM=SVM.predict(textml)
    prediction_classSVM=predtextSVM[0]

    #Multinomial NB
    
    predtextNB=NB.predict(textml)
    prediction_classNB=predtextSVM[0]

    prediction_votes=[prediction_classLSTM,prediction_classLR,prediction_classSVM,prediction_classNB]
    models=["LSTM","LR","SVM","NB"]
    votes_df=pd.DataFrame()
    votes_df['models']=models
    votes_df['predictions']=prediction_votes
    votes_df["predictions"] = votes_df['predictions'].replace({1:'Fake', 0:'True'})
    votes_df
    fake_vote=0
    true_vote=0

    for i in range(0,4):
        if prediction_votes[i]==1:
            fake_vote=fake_vote+1
            continue
        elif prediction_votes[i]==0:
            true_vote=true_vote+1
    
    if true_vote>fake_vote:
        prediction_class=0
    elif fake_vote>true_vote:
        prediction_class=1
    elif fake_vote==true_vote and (prediction_classLR+prediction_classSVM==2):
        prediction_class=1
    else:
        prediction_class=0


    

        
        


    if prediction_class == 0:
        st.success('This is not a fake news')
    if prediction_class == 1:
        st.warning('This is a fake news')

        
