import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle
from flask import Flask, render_template, request, jsonify
import datetime
from googlesearch import search


with open("intents.json") as file:
    data = json.load(file)

# Extract data 
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # word stemming 
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # Bag of words 
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

# clear default graph stack  
tensorflow.compat.v1.reset_default_graph()
# Develop NN model 
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Train & Save model 
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
            
    return np.array(bag)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message,words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.6:
            if tag == "about":
                response = "Firefly is an AI conversational chatbot application built in python and tensoflow by group * "
            else:
                for tg in data['intents']:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                response = random.choice(responses)
        else:
            # Response has low probability
            print(search(message, num_results=10))
            for i in search(message, num_results=10):
                print(i)
                response = i
        return str(response)
    return "Please say something!"

	
if __name__ == "__main__":
	app.run()