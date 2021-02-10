"""
Σε αυτή την δοκιμή θα ξεκινήσω αρχικά και θα πάρω 5000 δείγματα. και θα προσθέσω 1 layer.
"""
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt 
import numpy

with open('C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/log_acron.txt', 'r') as myfile:
    data=myfile.read()



data = data[:5000]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
#data
encoded = tokenizer.texts_to_sequences([data])[0]
encoded
vocab_size = len(tokenizer.word_index) + 1
vocab_size
print('Vocabulary Size: %d' % vocab_size)

# encode 2 words -> 1 word
sequences = list()
sequences
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
sequences    
print('Total Sequences: %d' % len(sequences))

max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_length-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(vocab_size, activation='softmax'))
                                                                                                             
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, validation_split=0.2, verbose=2, batch_size = 128)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

myfile.close()
print(model.summary())


def generate_seq(model, tokenizer, max_length, seed_text, n_words):
    in_text = seed_text
    #print("In_Text: ", in_text, "\n")
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        #print("Encoded: ", encoded, "\n")
        encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
        #print("Encoded: ", encoded, "\n")
        yhat = model.predict_classes(encoded, verbose=0)
        #print("Yhat: ", yhat, "\n")
        out_word = ''
        for word, index in tokenizer.word_index.items():
            #print("Word: ", word, " Index: ", index, "\n")
            if index == yhat:
                out_word = word
                #print("Equal\n")
                break
        in_text += ' ' + out_word
        #print("In_text: ", in_text, "\n")
    return in_text
 
tokenizer.word_index.items()
with open('C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/log_acron.txt', 'r') as myfile:
    data=myfile.read()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = list()

for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]

#yhat = model.predict(X, verbose=0)

print('Results: ')       
print ('\n')
print(generate_seq(model, tokenizer, max_length-1, 'acrtapp' , 3))  #ASBMTD
print(generate_seq(model, tokenizer, max_length-1, 'WVAPP', 3)) #wvapp
print(generate_seq(model, tokenizer, max_length-1, 'AACCPTD', 3)) #OCRTOFF
print(generate_seq(model, tokenizer, max_length-1, 'AVAL', 3)) #ORET
print(generate_seq(model, tokenizer, max_length-1, 'AINC', 3)) #WCINCFIL
print(generate_seq(model, tokenizer, max_length-1, 'WCINCFIL', 3)) #WCINCFIL
print(generate_seq(model, tokenizer, max_length-1, 'ASBMTD', 3)) #WHNDL
print(generate_seq(model, tokenizer, max_length-1, 'WHNDL', 3)) #WHNDL
myfile.close()
