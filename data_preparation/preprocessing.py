import os
import re

# Impostazione del backend di Keras su TensorFlow
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib

# Utilizzo del backend TkAgg per Matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
import keras

# Impostazione di un seed per la riproducibilità
keras.utils.set_random_seed(111)

# Percorso delle immagini
IMAGES_PATH = "input/images"

# Dimensioni desiderate per le immagini
IMAGE_SIZE = (224, 224)

# Lunghezza fissa consentita per qualsiasi sequenza
SEQ_LENGTH = 25


def load_captions_data(filename):
    # Carica i dati delle didascalie (testo) e li associa alle immagini corrispondenti.
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Nome immagine e didascalie sono separati da una tabulazione
            img_name, caption = line.split("\t")

            # Ogni immagine è ripetuta cinque volte per le cinque diverse didascalie.
            # Ogni nome immagine ha un suffisso `#(numero_didascalia)`
            img_name = img_name.split("#")[0]
            img_name = os.path.join(IMAGES_PATH, img_name.strip())

            # Rimuoviamo le didascalie troppo corte o troppo lunghe
            tokens = caption.strip().split()

            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # Aggiungiamo un token di inizio e uno di fine a ciascuna didascalia
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data


def train_val_split(caption_data, validation_size=0.2, test_size=0.05, shuffle=True):
    # 1. Ottenere la lista di tutti i nomi delle immagini
    all_images = list(caption_data.keys())

    # 2. Mescolare se necessario
    if shuffle:
        np.random.shuffle(all_images)

    train_keys, validation_keys = train_test_split(all_images, test_size=validation_size, random_state=42)
    validation_keys, test_keys = train_test_split(validation_keys, test_size=test_size, random_state=42)

    training_data = {img_name: caption_data[img_name] for img_name in train_keys}
    validation_data = {img_name: caption_data[img_name] for img_name in validation_keys}
    test_data = {img_name: caption_data[img_name] for img_name in test_keys}

    # Return the splits
    return training_data, validation_data, test_data


def regex():
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    return strip_chars


def custom_standardization(input_string):
    # Standardizzazione personalizzata del testo di input
    lowercase = tf.strings.lower(input_string)
    strip_chars = regex()
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


def decode_and_resize(img_path):
    # Decodifica e ridimensiona l'immagine dal percorso specificato
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def convert_to_lowercase():
    with open("input/text/token.txt", 'r') as f:
        lines = f.readlines()

    processed_lines = [line.lower() for line in lines]

    with open("input/text/token_processed.txt", 'w') as f:
        f.writelines(processed_lines)
