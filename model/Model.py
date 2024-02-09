import keras
import numpy as np
import tensorflow
from keras import layers
from keras.src.applications import efficientnet
from data_preparation.preprocessing import IMAGE_SIZE, decode_and_resize


class ImageCaptioningModel(keras.Model):
    def __init__(
            self,
            cnn_model,
            encoder,
            decoder,
            vectorization,
            num_captions_per_image=5,
            image_aug=None,
    ):
        super(ImageCaptioningModel, self).__init__()

        # Modello CNN per l'estrazione delle caratteristiche dell'immagine
        self.cnn_model = cnn_model

        # Encoder e Decoder per il modello di image captioning
        self.encoder = encoder
        self.decoder = decoder

        self.vectorization = vectorization

        # Tracker per la perdita e l'accuratezza
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")

        # Numero di didascalie per ogni immagine
        self.num_captions_per_image = num_captions_per_image

        # Augmentazione delle immagini
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        # Calcolo della perdita considerando la maschera
        loss = self.loss(y_true, y_pred)
        mask = tensorflow.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tensorflow.reduce_sum(loss) / tensorflow.reduce_sum(mask)

    @staticmethod
    def calculate_accuracy(y_true, y_pred, mask):
        # Calcolo dell'accuratezza considerando la maschera
        accuracy = tensorflow.equal(y_true, tensorflow.argmax(y_pred, axis=2))
        accuracy = tensorflow.math.logical_and(mask, accuracy)
        accuracy = tensorflow.cast(accuracy, dtype=tensorflow.float32)
        mask = tensorflow.cast(mask, dtype=tensorflow.float32)
        return tensorflow.reduce_sum(accuracy) / tensorflow.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        # Calcolo della perdita e dell'accuratezza per una didascalia
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tensorflow.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        # Passo di addestramento
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # Applicazione dell'aumentazione delle immagini se presente
        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # 1. Estrazione delle caratteristiche dell'immagine
        img_embed = self.cnn_model(batch_img)

        # 2. Passaggio di ciascuna delle cinque didascalie al decoder
        # insieme agli output dell'encoder e calcolo della perdita e dell'accuratezza
        # per ciascuna didascalia.
        for i in range(self.num_captions_per_image):
            with tensorflow.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # 3. Aggiornamento della perdita e dell'accuratezza
                batch_loss += loss
                batch_acc += acc

            # 4. Ottenimento della lista di tutti i pesi addestrabili
            train_vars = (
                    self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Ottenimento dei gradienti
            grads = tape.gradient(loss, train_vars)

            # 6. Aggiornamento dei pesi addestrabili
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Aggiornamento dei tracker
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Restituzione dei valori di perdita e accuratezza
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def test_step(self, batch_data):
        # Passo di valutazione
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Estrazione delle caratteristiche dell'immagine
        img_embed = self.cnn_model(batch_img)

        # 2. Passaggio di ciascuna delle cinque didascalie al decoder
        # insieme agli output dell'encoder e calcolo della perdita e dell'accuratezza
        # per ciascuna didascalia.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Aggiornamento della perdita e dell'accuratezza del batch
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Aggiornamento dei tracker
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Restituzione dei valori di perdita e accuratezza
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def call(self, image_path):
        img = decode_and_resize(image_path)

        img = tensorflow.expand_dims(img, 0)

        # Passa l'immagine alla CNN
        img = self.cnn_model(img)

        # Passa le caratteristiche dell'immagine all'encoder del Transformer
        encoded_img = self.encoder(img, training=False)

        # Ottieni il vocabolario dal layer di TextVectorization
        vocab = self.vectorization.get_vocabulary()
        # Crea un dizionario per la ricerca inversa del vocabolario
        index_lookup = dict(zip(range(len(vocab)), vocab))

        # Genera la didascalia usando il decoder del Transformer
        decoded_caption = "<start> "
        for i in range(24):
            tokenized_caption = self.vectorization([decoded_caption])[:, :-1]
            mask = tensorflow.math.not_equal(tokenized_caption, 0)
            predictions = self.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()

        return decoded_caption

    @property
    def metrics(self):
        # Lista delle metriche per consentire la chiamata automatica di `reset_states()`
        return [self.loss_tracker, self.acc_tracker]


def get_cnn_model():
    # Funzione per ottenere un modello CNN basato su EfficientNetB0
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # Congelamento dell'estrattore di caratteristiche
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model
