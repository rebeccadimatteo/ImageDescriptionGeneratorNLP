import os
from data_preparation.preprocessing import load_captions_data, train_val_split, custom_standardization, \
    decode_and_resize, convert_to_lowercase
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
import nltk
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

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


# Calculates BLEU score of predictions
def BLEU_score(actual, predicted):
    # Standardizing the actual captions
    processed_actual = []
    for i in actual:
        cap = [INDEX_LOOKUP[x] for x in vectorization(i).numpy() if INDEX_LOOKUP[x] != '']
        cap = ' '.join(cap)
        processed_actual.append(cap)

    split_processed_actual = [sentence.split() for sentence in processed_actual]
    #predicted = [sentence.split() for sentence in predicted]
    split_predicted = predicted.split()

    # Calculating the BLEU score by comparing the predicted caption with five actual captions.
    b1 = corpus_bleu([split_processed_actual], [split_predicted], weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu([split_processed_actual], [split_predicted], weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu([split_processed_actual], [split_predicted], weights=(0.3, 0.3, 0.3, 0))
    b4 = corpus_bleu([split_processed_actual], [split_predicted], weights=(0.25, 0.25, 0.25, 0.25))

    return [
        round(b1, 5),
        round(b2, 5),
        round(b3, 5),
        round(b4, 5)
    ]

def ROUGE_score(actual, predicted):
    rouge = Rouge()

    processed_actual = []
    for i in actual:
        cap = [INDEX_LOOKUP[x] for x in vectorization(i).numpy() if INDEX_LOOKUP[x] != '']
        cap = ' '.join(cap)
        processed_actual.append(cap)

    best_score = 0
    metrics = []
    for i in processed_actual:
        scores = rouge.get_scores(i, predicted)
        rouge_1_f_score = scores[0]['rouge-1']['f']
        if rouge_1_f_score > best_score:
            best_score = rouge_1_f_score
            metrics = scores

    #return metrics

    return [
        round(metrics[0]['rouge-1']['f'], 5),
        round(metrics[0]['rouge-2']['f'], 5),
        round(metrics[0]['rouge-l']['f'], 5)
    ]#restituiamo solo la media armonica tra precision e recall

def METEOR_score(actual, predicted):
    processed_actual = []

    for i in actual:
        cap = [INDEX_LOOKUP[x] for x in vectorization(i).numpy() if INDEX_LOOKUP[x] != '']
        cap = ' '.join(cap)
        processed_actual.append(cap)

    split_processed_actual = [sentence.split() for sentence in processed_actual]
    split_predicted = predicted.split()

    score = meteor_score(split_processed_actual, split_predicted)
    return round(score, 5)

def visualization(data, model, evaluator, num_of_images):
    keys = list(data.keys())  # List of all test images
    images = [np.random.choice(keys) for i in range(num_of_images)]  # Randomly selected images

    scores = []
    for filename in images:
        actual_cap = data[filename]
        actual_cap = [x.replace("<start> ", "") for x in actual_cap]  # Removing the start token
        actual_cap = [x.replace(" <end>", "") for x in actual_cap]  # Removing the end token

        predicted_cap = model(filename)
        # Getting the bleu score
        scores.append(evaluator(actual_cap, predicted_cap))

    return scores

def evaluation(data, model, num_of_images):
    scores_bleu = visualization(data, model, BLEU_score, num_of_images)
    scores_rouge = visualization(data, model, ROUGE_score, num_of_images)
    scores_meteor = visualization(data, model, METEOR_score, num_of_images)

    mean_bleu1 = mean_bleu2 = mean_bleu3 = mean_bleu4 = 0
    for i in range(len(scores_bleu)):
        mean_bleu1 += scores_bleu[i][0]
        mean_bleu2 += scores_bleu[i][1]
        mean_bleu3 += scores_bleu[i][2]
        mean_bleu4 += scores_bleu[i][3]

    mean_bleu1 = mean_bleu1 / len(scores_bleu)
    mean_bleu2 = mean_bleu2 / len(scores_bleu)
    mean_bleu3 = mean_bleu3 / len(scores_bleu)
    mean_bleu4 = mean_bleu4 / len(scores_bleu)


    mean_meteor = np.mean(scores_meteor)


    mean_rouge1 = mean_rouge2 = mean_rougel = 0
    for i in range(len(scores_rouge)):
        mean_rouge1 += scores_rouge[i][0]
        mean_rouge2 += scores_rouge[i][1]
        mean_rougel += scores_rouge[i][2]

    mean_rouge1 = mean_rouge1/len(scores_rouge)
    mean_rouge2 = mean_rouge2/len(scores_rouge)
    mean_rougel = mean_rougel/len(scores_rouge)


    return [mean_bleu1, mean_bleu2, mean_bleu3, mean_bleu4, mean_meteor, mean_rouge1, mean_rouge2, mean_rougel]

if __name__ == '__main__':
    convert_to_lowercase()
    # Carica il dataset
    captions_mapping, text_data = load_captions_data("input/text/token_processed.txt")

    # Suddivide il dataset in set di addestramento e validazione
    train_data, valid_data, test_data = train_val_split(captions_mapping)
    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(valid_data))
    print("Number of test samples: ", len(test_data))

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    # Passa la lista di immagini e la lista delle didascalie corrispondenti
    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))

    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

    # Data augmentation per le immagini
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder,
        vectorization=vectorization,
        image_aug=image_augmentation
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

    # Ottieni la lista dei nomi delle immagini nel set di dati di validazione
    #valid_images = list(valid_data.keys())

    vocab = vectorization.get_vocabulary()
    INDEX_LOOKUP = dict(zip(range(len(vocab)), vocab))
    MAX_DECODED_SENTENCE_LENGTH = SEQ_LENGTH - 1
    test_images = list(test_data.keys())

    scores = evaluation(test_data, caption_model, 20)

    print("BLEU-1: ", scores[0])
    print("BLEU-2: ", scores[1])
    print("BLEU-3: ", scores[2])
    print("BLEU-4: ", scores[3])
    print("METEOR: ", scores[4])
    print("ROUGE-1: ", scores[5])
    print("ROUGE-2: ", scores[6])
    print("ROUGE-L: ", scores[7])