from keras import layers
import tensorflow
from model.PositionalEmbedding import PositionalEmbedding


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        # Parametri del blocco Transformer Decoder
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        # Livelli del blocco Transformer Decoder
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        # Parametri per gli embedding
        EMBED_DIM = 512
        SEQ_LENGTH = 25
        VOCAB_SIZE = 10000

        # Layer di embedding posizionale
        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )

        # Output layer
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        # Dropout layers
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)

        # Indica che il layer supporta la masking
        self.supports_masking = True

    # Metodo di chiamata per il blocco Transformer Decoder
    def call(self, inputs, encoder_outputs, training, mask=None):
        # Applicazione dell'embedding posizionale
        inputs = self.embedding(inputs)

        # Creazione di una maschera causale per l'attenzione
        causal_mask = self.get_causal_attention_mask(inputs)

        # Creazione di una maschera combinata se fornita una maschera esterna
        if mask is not None:
            padding_mask = tensorflow.cast(mask[:, :, tensorflow.newaxis], dtype=tensorflow.int32)
            combined_mask = tensorflow.cast(mask[:, tensorflow.newaxis, :], dtype=tensorflow.int32)
            combined_mask = tensorflow.minimum(combined_mask, causal_mask)

        # Prima attenzione multi-testa
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Seconda attenzione multi-testa con encoder_outputs
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Feedforward neural network
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        # Aggiunta dell'output del feedforward all'output del blocco precedente
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)

        # Output finale attraverso il layer softmax
        preds = self.out(ffn_out)
        return preds

    # Metodo per ottenere una maschera causale per l'attenzione
    def get_causal_attention_mask(self, inputs):
        input_shape = tensorflow.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]

        # Creazione di una maschera causale
        i = tensorflow.range(sequence_length)[:, tensorflow.newaxis]
        j = tensorflow.range(sequence_length)
        mask = tensorflow.cast(i >= j, dtype="int32")
        mask = tensorflow.reshape(mask, (1, input_shape[1], input_shape[1]))

        # Espansione della maschera per il numero di batch
        mult = tensorflow.concat(
            [
                tensorflow.expand_dims(batch_size, -1),
                tensorflow.constant([1, 1], dtype=tensorflow.int32),
            ],
            axis=0,
        )
        return tensorflow.tile(mask, mult)
