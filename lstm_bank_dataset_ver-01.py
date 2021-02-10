from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout,TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt 
import numpy

with open('C:/Users/user/Desktop/ΔΙΠΛΩΜΑΤΙΚΗ/ΔΙΠΛΩΜΑΤΙΚΗ_2/log_acron.txt', 'r') as myfile:
    data=myfile.read()

myfile.close()

data = data[:80000]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data]) # Μετατροπή του text σε λίστα και fit.
#data
encoded = tokenizer.texts_to_sequences([data])[0] # Μετατροπή σε λίστα αριθμών.
#encoded
vocab_size = len(tokenizer.word_index) + 1 #word_index: A dictionary of words and their uniquely assigned integers.
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
print("Max length: ", max_length, "\n")

sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('Max Sequence Length: %d' % max_length)

sequences = array(sequences)
X, y = sequences[:,:-1],sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_length-1)) # Άλλαξα το 50 σε 32
model.add(LSTM(50))
model.add(Dropout(0.1))
model.add(Dense(vocab_size, activation='softmax'))

early_stopping = EarlyStopping(monitor='val_loss', patience=42)                                                                                                             

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=500, validation_split=0.2, verbose=2, batch_size = 128)

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
 


print('Results: ')   
print ('\n')
print(generate_seq(model, tokenizer, max_length-1, 'ACRTAPP' , 3)) # WCMPLT Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'WVAPP', 3)) # WVAPP Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'ORET', 3)) # WVAPP Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'AVAL', 3)) # WVAPP Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'AINC', 3)) # WCINCFIL Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'WCINCFIL', 3)) # WCINCFIL Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'ASBMTD', 3)) # WHNDL Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'WHNDL', 3)) # WCMPLT Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'OCRTD', 3)) # OSNTMLON Σωστό. Για 50.000 δείγματα το προβλέπει λάθος. Το σωστό είναι OSNTMLON
print(generate_seq(model, tokenizer, max_length-1, 'OCRTOFF', 3)) # OCRTD Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'ACNCPT', 3)) # WCMPLT Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'WCMPLT', 3)) # WCMPLT Σωστό
print(generate_seq(model, tokenizer, max_length-1, 'OSNTMLON', 3)) # WCMPLT Σωστό
