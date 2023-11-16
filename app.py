import pickle
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import spacy
import pandas as pd
import snowballstemmer
stemmer = snowballstemmer.stemmer('lithuanian')
# nlp = spacy.load("lt_core_news_lg")
nlp = spacy.load("lt_core_news_md")
stopwordsall = pd.read_csv('stopwords.csv')
stopwords = stopwordsall['stop_words'].tolist()
stop_words_set = set(stopwords)
nb = pickle.load(open('naivebmodel.pkl', 'rb'))
mlp = pickle.load(open('mlpmodel.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
vectorizerI = pickle.load(open('vetorizerI.pkl', 'rb'))
def sentence_corection(sentence):
    # Sentence is divided in tokens and they are processed to that the model would understand it better
    document = re.sub(r'\W', ' ', str(sentence))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Converting to Lowercase
    document = document.lower()
    document = re.sub("\d+", "", document)
    # Stop words filter
    document = document.split()
    document = [word for word in document if not word in stop_words_set]
    # Stemming
    document = ' '.join(document)
    document = nlp(document)
    document = [token.orth_ for token in document]
    document = stemmer.stemWords(document)
    document = ' '.join(document)
    return document
grasinantys_sakiniai = ['Negrasinantis', 'Grasinantis']
app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/process', methods=['POST'])
def process():
    input_text = request.get_json()
    correct_sentence = sentence_corection(input_text['data'])  # Example processing, 
    predictionOnMLP = mlp.predict(vectorizerI.transform([correct_sentence]))
    predictionOnNB = nb.predict(vectorizer.transform([correct_sentence]))
    response = {'NB': f"{grasinantys_sakiniai[predictionOnNB[0]]}", 'MLP': f"{grasinantys_sakiniai[predictionOnMLP[0]]}"}
    return jsonify(response)
if __name__ == '__main__':
    app.run()