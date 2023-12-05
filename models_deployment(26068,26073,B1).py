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
# from nltk.stem.porter import *
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model,model_from_json
from PIL import Image

wordnet = WordNetLemmatizer()

#Cleaning input data 
stops = set(stopwords.words("english"))
stop=stops


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
    text = [w for w in text if not w in stop]
    text=[wordnet.lemmatize(w) for w in text]
    text = " ".join(text)
    return text

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100   
GLOVE_DIR = "C:\\Users\\sah-1\\SML PROJECT\\glove.6B.100d.txt"
#Tokenizer
tokenizer_file="C:\\Users\\sah-1\\SML PROJECT\\tokenizer.pickle"

with open(tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)


# LSTM=load_model("C:\\Users\\sah-1\\SML PROJECT\\biLSTM.h5")
# load json and create model
json_file = open("C:\\Users\\sah-1\\SML PROJECT\\LSTM_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
LSTM = model_from_json(loaded_model_json)
# load weights into new model
LSTM.load_weights("C:\\Users\\sah-1\\SML PROJECT\\model_weights.h5")

filename="C:\\Users\\sah-1\\SML PROJECT\\best_LR_model.sav"
LR = pickle.load(open(filename, 'rb'))
filename="C:\\Users\\sah-1\\SML PROJECT\\SVM_model.sav"
SVM=pickle.load(open(filename, 'rb'))
filename="C:\\Users\\sah-1\\SML PROJECT\\NB_model.sav"
NB=pickle.load(open(filename, 'rb'))

#load the model from disk
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

# add_bg_from_local("C:\\Users\\sah-1\\OneDrive\\Pictures\\SML PROJECT\\bg5.jpg")


image=Image.open("C:\\Users\\sah-1\\Downloads\\project_logo.png")
st.columns(3)[1].image(image)
st.title('Whatsapp COVID-19 Fake News Classification app ')
st.subheader("Curbing Misinformation,One Message at a time ")

sentence = st.text_area("Enter your COVID related news content message  here", "Some news",height=200)
predict_btt = st.button("predict")

if predict_btt:
    clean_text = []
    #LSTM model
    clean_text = []
    i=cleaninput(sentence)
    clean_text.append(i)
    sequences = tokenizer.texts_to_sequences(clean_text)
    data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
    # text=[cleaninput(sentence)]
    # token_text=[tokenizer.texts_to_sequences(words) for words in text]
    # embedded_text=pad_sequences(token_text,padding='post',maxlen=MAX_SEQUENCE_LENGTH)
    prediction_classLSTM=LSTM.predict(data).argmax(axis=-1)[0]
    # prediction_classLSTM=predtextLSTM[0][0]

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

        
