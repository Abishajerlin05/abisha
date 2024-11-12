import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK data
nltk.download('popular', quiet=True)
nltk.download('punkt')
nltk.download('wordnet')

# Read the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenize the corpus
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Preprocessing function
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting input and response
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """Return a greeting response if user input contains a greeting"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Generating response based on cosine similarity
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.remove(user_response)
    return robo_response

# Flask route to serve the main page
@app.route("/")
def index():
    return render_template("index.html")

# Flask route to handle user message and chatbot response
@app.route("/get_response", methods=["POST"])
def get_response():
    user_message = request.form["message"]
    if user_message.lower() == 'bye':
        return "ROBO: Bye! Take care.."
    
    if user_message.lower() == 'thanks' or user_message.lower() == 'thank you':
        return "ROBO: You are welcome.."
    
    if greeting(user_message) is not None:
        return "ROBO: " + greeting(user_message)
    else:
        return "ROBO: " + response(user_message)

if __name__ == "__main__":
    app.run(debug=True)
