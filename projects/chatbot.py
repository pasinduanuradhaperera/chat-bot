import random
import json
import pickle
import numpy as np
from gtts import gTTS
import os

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('intents.json').read())

# Load preprocessed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = load_model('chatbot_model.model')

# Function to preprocess a sentence into a bag of words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class/intent of a sentence
def predict_class(sentence):
    bow = clean_up_sentence(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a random response for a predicted intent

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = None

    for intent in list_of_intents:
        if intent['intent'] == tag:
            result = random.choice(intent['responses'])
            break

    if result is None:
        result = "I'm sorry, I didn't understand that."
    else:
        result = random.choice(intent['responses'])  # Choose a random response from the matched intent

    return result

def return1():
    return "\nMandatory Rules:\n1.  Programming Language: Participants must use \"Python\" as the programming language for their solutions.\n2.  Application Type: The application should be a Console Application. Graphical User Interfaces (GUIs) are not allowed.\n3.  Console Communication: All communication with the participants' programs should occur through the console (standard input and output).\n4.  Uniqueness: Each participant's application must be unique. Code similarity among submissions will be scrutinized.\n5.  Original Work: Participants must create their applications from scratch. Code taken from online sources, tutorials, courses, or YouTube videos is not allowed.\n6.  Database Connection: Database connectivity is not required for this competition. Participants should not attempt to connect to a database.\n7.  Submission Format: Participants are required to submit their solutions as .py files within the specified time period. Other file formats will be examined and allowed only if necessary.\n\nConsiderations and Scoring:\n---------------------------\n1.  Code Clarity: Clear and well-documented code with comments and proper formatting will be rewarded with extra points.\n2.  Error Handling: Effective error handling, including try-except blocks, will earn participants extra points for robustness.\n3.  Library Usage: While libraries are allowed, participants who implement functionalities manually (without relying on libraries) will receive extra points.\n4.  Creativity: Creative solutions or unique problem-solving approaches will be recognized and awarded extra marks."
def return2():
    return "Google Doc Link: https://docs.google.com/presentation/d/1ytzHl36rG7i6PmzbjfRaJeguDztamO46D2lDYetQn0Q/edit?usp=sharing"

def speak_text(text):
    # Generate speech from the text
    tts = gTTS(text, lang='en')  # 'en' for English, you can change the language code as needed

    # Save the generated speech to a temporary MP3 file
    tts.save('temp.mp3')

    # Play the temporary MP3 file (requires an external player like 'afplay' on macOS)
    os.system('afplay temp.mp3')

    # Remove the temporary MP3 file
    os.remove('temp.mp3')

# Start the conversation

print('ChatBot: Hi! Welcome To Hack and Code')
speak_text('Hi! Welcome To Hack and Code')

while True:
    message = input("You: ")
    # Predict the intent of the user message
    if(message == 'exit'):
        break
    elif(message == 'rules' or message =='Rules'):
        message = 'tell me rules'
    ints = predict_class(message)
    # Get a response based on the predicted intent
    response = get_response(ints, intents)

    print("ChatBot:", response)
    if(response != return2() and response !=return1()):
        # Generate speech from the response
        speak_text(response)
    

print('😊 thanks for chat with me!!')
speak_text('thanks for chat with me')
exit()
