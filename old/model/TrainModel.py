import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, RepeatVector
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle


class ImageCaptioningModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_image_features(self, file_path='../processed/features.npy'):
        features = np.load(file_path, allow_pickle=True)
        image_names = features['image_name']
        features_df = pd.DataFrame(features['features'].tolist(),
                                   columns=[f'feature_{i}' for i in range(features['features'].shape[1])])
        features_df['image_name'] = image_names
        return features_df

    def load_captions(self, file_path='../processed/dataset.csv'):
        captions_df = pd.read_csv(file_path, delimiter='|')
        return captions_df

    def merge_data(self, features_df, captions_df):
        return pd.merge(features_df, captions_df, on='image_name')

    def prepare_train_test_data(self, data, test_size=0.2, random_state=42):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data

    def prepare_input_features(self, train_data, test_data):
        X_train = np.array(train_data.iloc[:, :-2])
        X_test = np.array(test_data.iloc[:, :-2])
        return X_train, X_test

    def prepare_output_descriptions(self, train_data, test_data, max_words=5000):
        y_train = train_data['comment;']
        y_test = test_data['comment;']

        start_token = '<START>'
        end_token = '<END>'
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        train_descriptions = [f"{start_token} {desc} {end_token}" for desc in y_train]
        test_descriptions = [f"{start_token} {desc} {end_token}" for desc in y_test]

        self.tokenizer.fit_on_texts(train_descriptions)
        vocab_size = len(self.tokenizer.word_index) + 1

        y_train_seq = self.tokenizer.texts_to_sequences(train_descriptions)
        y_test_seq = self.tokenizer.texts_to_sequences(test_descriptions)

        max_len = max(len(seq) for seq in y_train_seq)
        y_train_pad = pad_sequences(y_train_seq, maxlen=max_len, padding='post')
        y_test_pad = pad_sequences(y_test_seq, maxlen=max_len, padding='post')

        return y_train_pad, y_test_pad, self.tokenizer, vocab_size

    def build_model(self, input_shape, max_len, vocab_size):
        model = Sequential()
        model.add(Dense(256, input_shape=input_shape, activation='relu'))
        model.add(RepeatVector(max_len))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def train_model(self, X_train, y_train_pad, epochs=100, batch_size=64, validation_split=0.3):
        self.model.fit(X_train, y_train_pad, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def save_model(self, model_file='saved/trained_model.keras', tokenizer_file='saved/tokenizer.pickle'):
        self.model.save(model_file)
        with open(tokenizer_file, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    i_model = ImageCaptioningModel()

    print("Loading features")
    features = i_model.load_image_features()

    print("Loading captions")
    captions = i_model.load_captions()

    print("Merging data")
    data = i_model.merge_data(features, captions)

    print("Split data into train and test")
    train_data, test_data = i_model.prepare_train_test_data(data)
    X_train, X_test = i_model.prepare_input_features(train_data, test_data)
    y_train_pad, y_test_pad, tokenizer, vocab_size = i_model.prepare_output_descriptions(train_data, test_data)

    print("Build model")
    i_model.model = i_model.build_model(input_shape=(X_train.shape[1],), max_len=y_train_pad.shape[1], vocab_size=vocab_size)

    print("Train model")
    i_model.train_model(X_train, y_train_pad)

    print("Save model")
    i_model.save_model()


