import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import streamlit as st


ps = PorterStemmer()


st.title('Spam Classifier')
st.subheader('Enter your message:')
input_sms = st.text_input(label='SMS/EMAIL')

def transform_text(text):
  text = text.lower() # converting text to lower case
  text = nltk.word_tokenize(text) # splitting words form sentences
  
  y = []
  for i in text:   
    if i.isalnum():    # removing special characters
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:   # removing stopwords and punctuation
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
   
  text = y[:]
  y.clear()

  for i in text:             # stemming
     y.append(ps.stem(i))
  return y


tfid = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))



if st.button('Predict'):

  # preprocessing input data
  transform_sms = transform_text(input_sms)
  # vectorize
  vectorized_sms = tfid.transform(transform_sms)
  # predict
  result = model.predict(vectorized_sms)[0]
  # Display
  if result == 1:
    st.write('Spam')
  else:
    st.write('Not Spam')  
