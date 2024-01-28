from keras import layers
import tensorflow


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        # Creazione del layer di embedding per i token
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )

        # Creazione del layer di embedding per le posizioni
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

        # Memorizzazione dei parametri come attributi dell'istanza
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Calcolo della scala per gli embedding
        self.embed_scale = tensorflow.math.sqrt(tensorflow.cast(embed_dim, tensorflow.float32))

    def call(self, inputs, **kwargs):
        # Calcolo della lunghezza della sequenza di input
        length = tensorflow.shape(inputs)[-1]

        # Generazione di un vettore di posizioni da 0 a length-1
        positions = tensorflow.range(start=0, limit=length, delta=1)

        # Calcolo degli embedding per i token
        embedded_tokens = self.token_embeddings(inputs)

        # Moltiplicazione per la scala degli embedding
        embedded_tokens = embedded_tokens * self.embed_scale

        # Calcolo degli embedding per le posizioni
        embedded_positions = self.position_embeddings(positions)

        # Somma degli embedding di token e posizioni
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        # Creazione di una maschera booleana basata sugli input
        return tensorflow.math.not_equal(inputs, 0)
