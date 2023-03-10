from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

app = Flask(__name__, template_folder='templates')
ps = PorterStemmer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tfid = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))

    input_sms = request.form['input_sms']
    transform_sms = transform_text(input_sms)

    vectorized_sms = tfid.transform(transform_sms)

    result = model.predict(vectorized_sms)[0]

    if result == 1:
        output = 'Spam'
    else:
        output = 'Not Spam'

    return render_template('index.html', prediction_text='{}'.format(output))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return y

if __name__ == '__main__':
    app.run(debug=True)
