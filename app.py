from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder='templates')
ps = PorterStemmer()

# Load the trained model and vectorizer from pickle files during startup
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_sms = request.form['input_sms']
        if not input_sms:
            raise ValueError('Please enter a SMS message')
        
        # Preprocess the input SMS text
        text = transform_text(input_sms)
        
        # Transform the preprocessed text using the loaded vectorizer
        vectorized_text = tfidf.transform(text)
        
        # Predict the class label using the loaded model
        predicted_class = model.predict(vectorized_text)[0]
        
        # Map the predicted class label to a human-readable output message
        output_message = 'Spam' if predicted_class == 1 else 'Not Spam'
        
        return render_template('index.html', prediction_text=output_message)
    
    except ValueError as e:
        error_message = str(e)
        return render_template('index.html', error=error_message)

def transform_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and punctuation, and stem the words
    stopwords_set = set(stopwords.words('english'))
    punctuation_set = set(string.punctuation)
    stemmed_tokens = [ps.stem(token) for token in tokens 
                      if token not in stopwords_set and token not in punctuation_set]
    
    return stemmed_tokens

if __name__ == '__main__':
    app.run()
