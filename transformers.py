import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D 
from keras.saving import register_keras_serializable



import tensorflow as tf
from tensorflow.keras import layers

@register_keras_serializable(package="Custom")
class ImagePositionalEmbedding(layers.Layer):
    def __init__(self, num_patches, d_model,**k):
        super().__init__(**k)
        self.num_patches = num_patches
        self.d_model = d_model
        
        # Learnable position embeddings
        self.pos_embedding = layers.Embedding(num_patches,d_model)

    def call(self, x):
        # x shape: (batch_size, 49, d_model)
        
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.pos_embedding(positions)  # (49, d_model)
        
        return x + pos_embeddings




@register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, d_model, max_len,**k):
        super().__init__(**k)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_len, d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions




@register_keras_serializable(package="Custom") 
class TransformerEncoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.3,**k):
        super().__init__(**k)
        self.att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x,training=False):
        attn_output = self.att(x, x,x)
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output,training=training)
        return self.norm2(out1 + ffn_output)


    



@register_keras_serializable(package="Custom")  # to save custom layer with model
class TransformerDecoder(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1,**k):
        super().__init__(**k)
        self.self_att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)
        self.enc_dec_att = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads)

        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.norm3 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output,dec_padding_mask):
        # Masked self-attention
        att1 = self.self_att(x, x, x, use_causal_mask=True,attention_mask=dec_padding_mask)
        att1 = self.dropout1(att1)
        out1 = self.norm1(x + att1)

        # Encoder–Decoder attention
        att2 = self.enc_dec_att(query=out1,value=enc_output,key=enc_output)
        att2 = self.dropout2(att2)
        out2 = self.norm2(out1 + att2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.norm3(out2 + ffn_output)




@register_keras_serializable(package="Custom")
class PaddingMask(layers.Layer):
    def call(self, x):
        mask = tf.not_equal(x, 100277)
        return mask[:, tf.newaxis, :]   # (batch,1,seq_len)                              # boolean
