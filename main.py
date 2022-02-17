from shutil import move
from typing import Iterator
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import wikipedia
import os
import speech_recognition as sr
import pyttsx3
from datetime import datetime
import winsound
import time
from nltk.stem.lancaster import LancasterStemmer
from pyfirmata import Arduino, util
#board = Arduino('COM3')
#iterator = util.Iterator(board)
#iterator.start();
#pin9 = board.get_pin('d:9:s')
#def move_servo(a):
#    pin9.write(a)
Listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)
newVoiceRate = 120
engine.setProperty('rate', newVoiceRate)
stemmer = LancasterStemmer()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["text"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["intent"])

        if intent["intent"] not in labels:
            labels.append(intent["intent"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

#def blink():
    move_servo(0)
    move_servo(60)
    time.sleep(1)
    move_servo(0)

def chat():
    frequency = 200  # Set Frequency To 200 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    while True:
        #with sr.Microphone() as source:
        #    voice = Listener.listen(source)
        
        inp = input(":")
        
        print(inp)
        if inp == "quit":  
                print("goodbye")
                engine.say("goodbye")
                engine.runAndWait()
                break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['intent'] == tag:
                responses = tg['responses']
                if(tg['intent'] == "TimeQuery"):
                    now = datetime.now()
                    current_time = now.strftime("%I:%M %p")   
                    engine.say(current_time) 
                    engine.runAndWait()
                    responses = [" ", " ", " "]
                if(tg['intent'] == "Search"):
                    wordz = inp.split()
                    
                    engine.say(wikipedia.summary(wordz[-1]))
                    engine.runAndWait();
                    responses = [" ", " ", " "]
        #blink()
        print(random.choice(responses))
        engine.say(random.choice(responses))
        engine.runAndWait()

chat()
