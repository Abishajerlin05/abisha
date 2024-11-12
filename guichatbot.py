import io
import random
import string
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext

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

# Chatbot GUI with Tkinter
def send_message():
    user_response = user_input.get()
    chat_window.insert(tk.END, "You: " + user_response + '\n')
    user_input.delete(0, tk.END)

    if user_response.lower() == 'bye':
        chat_window.insert(tk.END, "ROBO: Bye! Take care..\n")
        return

    if user_response.lower() == 'thanks' or user_response.lower() == 'thank you':
        chat_window.insert(tk.END, "ROBO: You are welcome..\n")
        return

    if greeting(user_response) is not None:
        chat_window.insert(tk.END, "ROBO: " + greeting(user_response) + '\n')
    else:
        robo_reply = response(user_response)
        chat_window.insert(tk.END, "ROBO: " + robo_reply + '\n')

# Create the main window
root = tk.Tk()
root.title("Robo Chatbot")
root.geometry("400x500")

# Create a scrollable text widget for displaying conversation
chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20, font=("Helvetica", 12))
chat_window.grid(row=0, column=0, padx=10, pady=10)
chat_window.config(state=tk.DISABLED)

# Create an entry widget for user input
user_input = tk.Entry(root, width=40, font=("Helvetica", 12))
user_input.grid(row=1, column=0, padx=10, pady=10)

# Create a send button
send_button = tk.Button(root, text="Send", width=10, font=("Helvetica", 12), command=send_message)
send_button.grid(row=2, column=0, pady=10)

# Run the GUI
root.mainloop()
