import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from PIL import Image
import pickle


# Funzione per ridimensionare l'immagine
def resize_image(input_path, target_size=(299, 299)):
    try:
        # Apre l'immagine e la ridimensiona
        img = Image.open(input_path)
        img = img.resize(target_size, resample=Image.LANCZOS)

        # Restituisce l'immagine ridimensionata
        return img
    except Exception as e:
        print(f"Errore nel ridimensionare l'immagine {input_path}: {str(e)}")
        return None


# Funzione per estrarre le caratteristiche da un'immagine
def extract_features(img):
    # Riduci le dimensioni dell'immagine per adattarle al modello VGG16
    img = img.resize((224, 224))

    # Converti l'immagine in un array numpy
    img_array = image.img_to_array(img)

    # Aggiungi una dimensione batch
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocessa l'immagine
    img_array = preprocess_input(img_array)

    # Estrai le caratteristiche usando il modello VGG16
    features = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).predict(img_array)

    # Riduci le dimensioni delle features a 25088
    selected_features = features.flatten()[:25088]

    return selected_features


def generate_description(features, max_len=52, temperature=1.0):
    try:
        # Carica il tokenizer
        with open('saved/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        if tokenizer is None:
            raise ValueError("Tokenizer not loaded successfully.")

        # Carica il modello addestrato
        model = load_model('saved/trained_model.keras')

        description = []

        # Aggiungi una dimensione batch (anche se è 1) per l'input del modello
        features = np.expand_dims(features, axis=0)

        # Carica il dizionario index_word dal tokenizer
        index_word = tokenizer.index_word

        # Aggiungi <START> alla descrizione iniziale
        description.append('<START>')

        for _ in range(max_len):
            # Fai una predizione con il modello
            predicted_sequence = model.predict(features)

            # Converti la sequenza predetta in un campione basato sulla temperatura
            predicted_sequence = np.asarray(predicted_sequence[0, -1, :]) / temperature
            normalized_probs = predicted_sequence / np.sum(predicted_sequence)
            predicted_word_index = np.random.choice(len(predicted_sequence), p=normalized_probs)

            # Verifica che l'indice predetto sia presente nel dizionario index_word inoltrato
            predicted_word = index_word.get(predicted_word_index, None)

            if predicted_word is not None:
                # Ignora i token <START> e <END>
                if predicted_word != '<START>' and predicted_word != '<END>':
                    # Aggiungi la parola alla descrizione
                    description.append(predicted_word)

                # Aggiorna le features con la parola predetta per il passo successivo
                features[0, -1] = predicted_word_index

                # Aggiungi la condizione per interrompere se è predetto '<END>'
                if predicted_word == '<END>':
                    break

        # Rimuovi eventuali token <START> e <END> residui
        description = [word for word in description if word not in ['<START>', '<END>']]

        return ' '.join(description)

    except Exception as e:
        print(f"Error during description generation: {str(e)}")
        return None


if __name__ == '__main__':
    # Specifica il percorso dell'immagine di input
    input_image_path = '../input/images/148284.jpg'
    target_size = (224, 224)  # Modificato per adattarsi alle dimensioni di input del modello VGG16

    # Ridimensiona l'immagine
    resized_image = resize_image(input_image_path, target_size)

    # Estrai le caratteristiche dall'immagine
    features = extract_features(resized_image)

    # Chiama la funzione per generare la descrizione
    description = generate_description(features, max_len=53, temperature=0.8)

    # Stampa la descrizione generata
    print("Descrizione generata:", description)
