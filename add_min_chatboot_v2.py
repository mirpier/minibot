#import pickle # per serializzare e deserializzare oggetti in Python
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
import nltk  # analisi linguistica e elaborare i dati di input dell’utente
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Inizializzazione (intens.json contiente una serie di pattern che utilizzeremo per recuperare la risposta più adeguata dell’utente)
words = []
classes = []
documents = []
ignore_words = ['?', '!']

df = pd.read_json("intents_min_2.json", orient="records")

for intent in df['intents']:
        
    for pattern in intent['patterns']:
   
        # tokenizzo ogni parola (divido tutte le parole in intents)
        w = nltk.word_tokenize(pattern)
        words.extend(w)

        # aggiungo all'array documents
        documents.append((w, intent['tag']))

        # aggiungo classi al nostro elenco
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer() # verbo camminare può apparire come cammina, camminò, camminando e così via. La forma canonica, camminare, è il lemma della parola ed è la forma di riferimento per cercare la parola all'interno di un dizionario

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

# """# **Addestramento**"""

# # Data preparation 
# input_net = []
# output_des = []
# output_empty = [0] * len(classes)
# for doc in documents:
#     # bag of words # quante volte ciao, ecc
#     bag = []
#     # lista di token
#     pattern_words = doc[0]
#     # lemmatizzazione dei token
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     # se la parola corrisponde inserisco 1, altrimenti 0
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     input_net.append([bag])
#     output_des.append([output_row])
#     #training.append([bag, output_row]) # questo è il training set


# # Creazione del modello
# model = Sequential()
# #model.add(Dense(128, input_shape=(len(input_net),), activation='relu'))
# model.add(Dense(128, input_shape=(len(bag),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# #model.add(Dense(len(output_des), activation='softmax'))
# model.add(Dense(len(output_row), activation='softmax'))

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# #fitting and saving the model

# input_net = np.array(input_net)
# output_des  = np.array(output_des)

# print (input_net)
# print(output_des)

# #TODO: inserireseconda dimenzione dinamica
# input_net = np.reshape(input_net, (input_net.shape[0], input_net.shape[2]))
# output_des = np.reshape(output_des, (output_des.shape[0], output_des.shape[2]))

# hist = model.fit(input_net, output_des, epochs=300, batch_size=5, verbose=1)
# model.save('chatbot_model.h5', hist)

# print("Modello creato!")

model = load_model('chatbot_model.h5')

# pre-elaborazione input utente
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# creazione bag of words
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# calcolo delle possibili risposte
def calcola_pred(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# restituzione della risposta
def getRisposta(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = np.random.choice(i['responses'])
            break
    return result


def dialoga(msg):
    print(msg)
    results = calcola_pred(msg, model)    
    response = getRisposta(results, df)
    return response
    
