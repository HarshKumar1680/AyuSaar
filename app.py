from flask import Flask, render_template, request, jsonify
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
import os
import time
import webbrowser
from threading import Timer

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('popular')
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

lemmatizer = WordNetLemmatizer()

# Load model and data files
try:
    model = load_model('model.h5')
    with open('data.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    with open('texts.pkl', 'rb') as file:
        words = pickle.load(file)
    with open('labels.pkl', 'rb') as file:
        classes = pickle.load(file)
except Exception as e:
    print(f"Error loading model or data files: {str(e)}")
    raise

# Add new global variable for metrics file
METRICS_FILE = 'metrics.json'

def clean_up_sentence(sentence):
    try:
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    except Exception as e:
        print(f"Error in clean_up_sentence: {str(e)}")
        return []

def bow(sentence, words, show_details=False):
    try:
        # tokenize the pattern
        sentence_words = clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print(f"found in bag: {w}")
        if show_details:
            print(f"Input sentence: {sentence}")
            print(f"Tokenized words: {sentence_words}")
            print(f"Bag of words: {bag}")
        return np.array(bag)
    except Exception as e:
        print(f"Error in bow: {str(e)}")
        return np.zeros(len(words))

def predict_class(sentence, model):
    try:
        # Convert to lowercase and strip whitespace
        sentence = sentence.lower().strip()
        
        # Direct pattern matching first
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Check for exact match
                if pattern.lower() == sentence:
                    return [{"intent": intent['tag'], "probability": "1.0"}]
                # Check if pattern is a substring of the sentence
                elif pattern.lower() in sentence:
                    return [{"intent": intent['tag'], "probability": "0.9"}]
        
        # If no direct match, use model prediction
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]), verbose=0)[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(f"Predicted intents: {return_list}")
        return return_list
    except Exception as e:
        print(f"Error in predict_class: {str(e)}")
        return []

def getResponse(ints, intents_json):
    try:
        if not ints:
            # Check for simple greetings manually
            user_input = request.args.get('msg', '').lower().strip()
            if user_input in ['hi', 'hello', 'hey', 'hii']:
                return random.choice(intents_json['intents'][0]['responses'])
            return "I'm not sure I understand. Could you please rephrase that?"
        
        tag = ints[0]['intent']
        print(f"Selected intent tag: {tag}")
        
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'].lower() == tag.lower():
                # Check if 'responses' or 'response' key exists
                if 'responses' in i:
                    result = random.choice(i['responses'])
                elif 'response' in i:
                    result = random.choice(i['response'])
                else:
                    continue
                print(f"Selected response: {result}")
                return result
        
        print(f"No matching intent found for tag: {tag}")
        return "I'm not sure I understand. Could you please rephrase that?"
    except Exception as e:
        print(f"Error in getResponse: {str(e)}")
        return "I'm having trouble understanding. Could you please try again?"

def chatbot_response(msg):
    if not msg or not isinstance(msg, str):
        return "Please provide a valid message."
    
    # Remove quotes if present and clean the message
    msg = msg.strip('"\'')
    msg = msg.lower().strip()
    
    # Handle common variations and typos
    msg = msg.replace("i'm", "i am")
    msg = msg.replace("i've", "i have")
    msg = msg.replace("i'll", "i will")
    msg = msg.replace("i'd", "i would")
    msg = msg.replace("i'm", "i am")
    msg = msg.replace("i've", "i have")
    msg = msg.replace("i'll", "i will")
    msg = msg.replace("i'd", "i would")
    
    print(f"Processing message: {msg}")
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

# Add new function to store metrics
def store_metrics(metrics):
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(metrics)
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Modify the /get endpoint to track response time and confidence
@app.route("/get")
def get_bot_response():
    try:
        userText = request.args.get('msg')
        if not userText:
            return "Please provide a message."
        start_time = time.time()
        response = chatbot_response(userText)
        end_time = time.time()
        response_time = end_time - start_time
        # Assume confidence is the probability from the first intent
        confidence = 0.0
        ints = predict_class(userText, model)
        if ints:
            confidence = float(ints[0].get('probability', 0.0))
        metrics = {
            'user_message': userText,
            'response': response,
            'response_time': response_time,
            'confidence': confidence
        }
        store_metrics(metrics)
        return jsonify({
            'response': response,
            'response_time': response_time,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error in get_bot_response: {str(e)}")
        return "An error occurred. Please try again."

# Add new endpoint for user satisfaction ratings
@app.route("/rate", methods=['POST'])
def rate_response():
    try:
        data = request.json
        rating = data.get('rating')
        if rating is None:
            return jsonify({'error': 'Rating is required'}), 400
        metrics = {
            'rating': rating,
            'timestamp': time.time()
        }
        store_metrics(metrics)
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in rate_response: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == "__main__":
    # Open the browser after a short delay
    Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000/')).start()
    app.run(debug=True)
