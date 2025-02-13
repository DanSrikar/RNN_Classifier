import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

st.title("Movie Review Sentiment Classifier using RNN")
word_index=imdb.get_word_index()


model=load_model('simple_rnn_imdb.h5')

def preprocessing(text):
    res=text.lower().split()
    mapp_res=[word_index.get(word,2)+3 for word in res]
    padded_review=sequence.pad_sequences([mapp_res],maxlen=500)
    return padded_review

def predict_sentiment(text):
    input=preprocessing(text)
    prediction=model.predict(input)
    if prediction[0][0]>0.5:
        return "Positive",prediction[0][0]
    else:
        return "Negative",prediction[0][0]
    


a=st.text_area("Enter your review")

if st.button("Classify"):
    sentiment,score=predict_sentiment(a)
    st.write(f"Sentiment : {sentiment}")
    st.write(f"Score : {score}")

else:
    st.write(f"Pls enter a review")
    

    
