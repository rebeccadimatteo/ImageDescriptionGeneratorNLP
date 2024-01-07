import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

# Carica le features delle immagini
features = np.load('features.npy', allow_pickle=True)

# Estrai solo le colonne necessarie da features
image_names = features['image_name']
features_df = pd.DataFrame(features['features'].tolist(), columns=[f'feature_{i}' for i in range(features['features'].shape[1])])
features_df['image_name'] = image_names

# Carica la descrizione delle immagini da un file CSV
captions_df = pd.read_csv('C:\\Users\\becca\\PycharmProjects\\ImageDescriptionGeneratorNLP\\results.csv', delimiter='|')

# Unisci le features con le descrizioni basate sulla colonna image_name
data = pd.merge(features_df, captions_df, on='image_name')

# Dividi i dati in set di addestramento e test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepara le features di input
X_train = np.array(train_data.iloc[:, :-2])  # Exclude 'image_name' and 'comment' columns
X_test = np.array(test_data.iloc[:, :-2])   # Exclude 'image_name' and 'comment' columns

# Prepara le descrizioni di output
y_train = train_data['comment;']
y_test = test_data['comment;']

## Tokenizza le descrizioni
max_words = 2000
start_token = '<START>'
end_token = '<END>'
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')  # Aggiungi filters per evitare la rimozione di ";"

# Aggiungi <START> e <END> a ogni descrizione
train_descriptions = [f"{start_token} {desc} {end_token}" for desc in y_train]
test_descriptions = [f"{start_token} {desc} {end_token}" for desc in y_test]

tokenizer.fit_on_texts(train_descriptions)  # Fit sulla colonna di addestramento
vocab_size = len(tokenizer.word_index) + 1

# Sequenze di parole
y_train_seq = tokenizer.texts_to_sequences(train_descriptions)
y_test_seq = tokenizer.texts_to_sequences(test_descriptions)

# Padding per ottenere sequenze della stessa lunghezza
max_len = max(len(seq) for seq in y_train_seq)  # Considera la lunghezza massima tra le sequenze
y_train_pad = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
y_test_pad = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

# Modello
model = Sequential()

# Aggiunge uno strato denso (fully connected) con 256 unità di output e funzione di attivazione ReLU
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))

# Aggiunge uno strato di RepeatVector, che replica i vettori di input per ottenere una sequenza di lunghezza specifica (max_len)
model.add(RepeatVector(max_len))

# Aggiunge uno strato LSTM con 256 unità di output, che restituisce le sequenze complete
model.add(LSTM(256, return_sequences=True))

# Aggiunge uno strato densamente connesso con un numero di unità di output pari alla dimensione del vocabolario e funzione di attivazione softmax
model.add(Dense(vocab_size, activation='softmax'))

# Cambia il tipo di loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Addestra il modello per un maggior numero di epoche
model.fit(X_train, y_train_pad, epochs=20, batch_size=64, validation_split=0.2)

# Salva il modello addestrato
model.save('trained_model.h5')

# Salva il tokenizzatore
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Carica il modello addestrato
loaded_model = load_model('trained_model.h5')

# Carica il tokenizzatore addestrato
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

