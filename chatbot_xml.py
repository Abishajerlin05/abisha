import io
import random
import string  # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET

nltk.download('popular', quiet=True)  # for downloading packages

# Reading in the corpus
with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
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
    if(req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


# XML setup: Create the root element
root = ET.Element("conversations")

flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while flag:
    user_response = input()
    user_response = user_response.lower()

    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                chatbot_reply = greeting(user_response)
                print("ROBO: " + chatbot_reply)
            else:
                chatbot_reply = response(user_response)
                print("ROBO: " + chatbot_reply)
                sent_tokens.remove(user_response)

            # Creating a new conversation entry
            conversation = ET.SubElement(root, "conversation")
            user_input_elem = ET.SubElement(conversation, "user_input")
            user_input_elem.text = user_response
            chatbot_response_elem = ET.SubElement(conversation, "chatbot_response")
            chatbot_response_elem.text = chatbot_reply

    else:
        flag = False
        print("ROBO: Bye! Take care..")
        # Adding exit conversation to XML
        conversation = ET.SubElement(root, "conversation")
        user_input_elem = ET.SubElement(conversation, "user_input")
        user_input_elem.text = 'bye'
        chatbot_response_elem = ET.SubElement(conversation, "chatbot_response")
        chatbot_response_elem.text = 'Bye! Take care..'

# Save the XML data to a file
tree = ET.ElementTree(root)
tree.write("chatbot_interactions.xml", encoding='utf-8', xml_declaration=True)
