from keras import layers


# Definizione di un blocco Transformer Encoder come sottoclasse di layers.Layer
class TransformerEncoderBlock(layers.Layer):

    # Costruttore della classe
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        # Parametri del blocco Transformer
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        # Livelli del blocco Transformer
        # Attenzione multi-testa con parametri specificati
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )

        # Layer di normalizzazione dopo la prima operazione di attenzione
        self.layernorm_1 = layers.LayerNormalization()

        # Layer di normalizzazione dopo l'output finale del blocco
        self.layernorm_2 = layers.LayerNormalization()

        # Rete densa con attivazione ReLU
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    # Metodo di chiamata per il blocco Transformer
    def call(self, inputs, training, mask=None):
        # Normalizzazione layer prima della rete densa
        inputs = self.layernorm_1(inputs)

        # Applicazione della rete densa con attivazione ReLU
        inputs = self.dense_1(inputs)

        # Applicazione dell'attenzione multi-testa
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )

        # Aggiunta dell'output dell'attenzione ai dati di input originali, normalizzazione layer e output del blocco Transformer
        out_1 = self.layernorm_2(inputs + attention_output_1)

        # Restituzione dell'output del blocco Transformer
        return out_1

