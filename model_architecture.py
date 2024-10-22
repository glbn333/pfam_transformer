import tensorflow as tf
from tensorflow.keras import layers

# Define a custom layer for a single Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # Multi-head attention layer
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # Layer normalization layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # Multi-head attention layer
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Layer normalization and residual connection
        return self.layernorm2(out1 + ffn_output)

# Define a custom layer for token and position embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # Token embedding layer
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        # Position embedding layer
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        # Generate position indices
        positions = tf.range(start=0, limit=maxlen, delta=1)
        # Generate position embeddings
        positions = self.pos_emb(positions)
        # Generate token embeddings
        x = self.token_emb(x)
        # Add token and position embeddings
        return x + positions

# Define a function to create the Transformer model
def create_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim, num_layers, num_classes):
    # Input layer
    inputs = layers.Input(shape=(maxlen,))
    # Token and position embedding layer
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    # Transformer blocks
    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    # Global average pooling layer
    x = layers.GlobalAveragePooling1D()(x)
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
