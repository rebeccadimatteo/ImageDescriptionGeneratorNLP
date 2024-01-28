import os
from data_preparation.preprocessing import load_captions_data, train_val_split, custom_standardization, \
    decode_and_resize
from model.LRSchedule import LRSchedule
from model.Model import ImageCaptioningModel, get_cnn_model
from keras import layers
from keras.layers import TextVectorization
import matplotlib
from trasformer.Decoder import TransformerDecoderBlock
from trasformer.Encoder import TransformerEncoderBlock
import numpy as np
import tensorflow as tf
import keras

# Impostazione del backend di Keras su TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Utilizzo del backend TkAgg per Matplotlib
matplotlib.use('TkAgg')

# Impostazione di un seed per la riproducibilità
keras.utils.set_random_seed(13)

# Percorso delle immagini
IMAGES_PATH = "input/images"

# Dimensioni desiderate per le immagini
IMAGE_SIZE = (299, 299)

# Dimensione del vocabolario
VOCAB_SIZE = 10000

# Lunghezza fissa consentita per qualsiasi sequenza
SEQ_LENGTH = 25

# Dimensione per gli embedding delle immagini e degli token
EMBED_DIM = 512

# Unità per layer nella rete feed-forward
FF_DIM = 512

# Altri parametri di addestramento
BATCH_SIZE = 64
EPOCHS = 1


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


def test_model(sample_img):
    path = sample_img
    sample_img = decode_and_resize(sample_img)

    # Passa l'immagine alla CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Passa le caratteristiche dell'immagine all'encoder del Transformer
    encoded_img = caption_model.encoder(img, training=False)

    # Genera la didascalia usando il decoder del Transformer
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption for ", path, ": ", decoded_caption)


if __name__ == '__main__':
    # Carica il dataset
    captions_mapping, text_data = load_captions_data("input/text/token.txt")

    # Suddivide il dataset in set di addestramento e validazione
    train_data, valid_data = train_val_split(captions_mapping)
    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(valid_data))

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    # Data augmentation per le immagini
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )

    # Passa la lista di immagini e la lista delle didascalie corrispondenti
    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))

    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder,
        image_aug=image_augmentation,
    )

    # Definizione della funzione di perdita
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=keras.losses.Reduction.AUTO,  # Impostare la riduzione su 'auto'
    )

    # Criteri di EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Creazione di un programma di apprendimento
    num_train_steps = len(train_dataset) * EPOCHS
    num_warmup_steps = num_train_steps // 15
    lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

    # Compila il modello
    caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

    # Addestra il modello
    caption_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=[early_stopping],
    )

    # Ottieni il vocabolario dal layer di TextVectorization
    vocab = vectorization.get_vocabulary()
    # Crea un dizionario per la ricerca inversa del vocabolario
    index_lookup = dict(zip(range(len(vocab)), vocab))
    # Determina la lunghezza massima per la generazione di didascalie
    max_decoded_sentence_length = SEQ_LENGTH - 1
    # Ottieni la lista dei nomi delle immagini nel set di dati di validazione
    valid_images = list(valid_data.keys())

    test_model(sample_img="input/test/1.jpg")
    test_model(sample_img="input/test/2.jpg")
    test_model(sample_img="input/test/3.jpg")
    test_model(sample_img="input/test/4.jpg")
    test_model(sample_img="input/test/5.jpg")
    test_model(sample_img="input/test/6.jpg")
    test_model(sample_img="input/test/7.jpg")
    test_model(sample_img="input/test/8.jpg")
    test_model(sample_img="input/test/9.jpg")
    test_model(sample_img="input/test/10.jpg")
