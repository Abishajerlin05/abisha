import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import numpy as np

# Load NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer

# Load the JSON data
with open('chatbot_data.json', 'r') as f:
    data = json.load(f)

# Initialize Lemmatizer
lemmer = WordNetLemmatizer()

# Tokenization and Lemmatization function
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Greeting function
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in data["greetings"]["inputs"]:
            return random.choice(data["greetings"]["responses"])
    return None

# Response generation function
def response(user_response):
    """Generate a response based on cosine similarity with knowledge base"""
    robo_response = ''
    
    # Prepare corpus for similarity comparison
    knowledge_sentences = [item["answer"] for item in data["knowledge"]]
    
    # Add the user response to the list (used for similarity comparison)
    knowledge_sentences.append(user_response)
    
    # Create a TfidfVectorizer to compare the user input with the knowledge base
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(knowledge_sentences)
    
    # Compute cosine similarity
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    # Find the most similar answer
    idx = vals.argsort()[0][-2]  # Get the index of the second most similar item
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    # If no good match found, return a default message
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = knowledge_sentences[idx]
    
    return robo_response


# Main loop
def chatbot():
    print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
    
    flag = True
    while flag:
        user_response = input()
        user_response = user_response.lower()
        
        if user_response != 'bye':
            if user_response in ['thanks', 'thank you']:
                flag = False
                print("ROBO: You are welcome..")
            else:
                greeting_response = greeting(user_response)
                if greeting_response:
                    print("ROBO: " + greeting_response)
                else:
                    print("ROBO: ", end="")
                    print(response(user_response))
        else:
            flag = False
            print("ROBO: Bye! Take care..")

if __name__ == "__main__":
    chatbot()
