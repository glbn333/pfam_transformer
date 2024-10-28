import tensorflow as tf
from keras import layers
import keras

# Define a custom layer for a single Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        # Multi-head attention layer
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        # Layer normalization layers
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        """
        Executes the forward pass of the transformer layer.

        Args:
            inputs (tf.Tensor): Input tensor to the transformer layer.
            training (bool, optional): Boolean flag indicating whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: Output tensor after applying multi-head attention, feed-forward network, and layer normalization.

        The method performs the following steps:
        1. Applies multi-head attention to the inputs.
        2. Applies dropout to the attention output if training is True.
        3. Adds the attention output to the original inputs and normalizes the result.
        4. Passes the normalized result through a feed-forward network.
        5. Applies dropout to the feed-forward network output if training is True.
        6. Adds the feed-forward network output to the normalized result and normalizes the final output.
        """
        # Multi-head attention layer
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(inputs=attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(inputs=ffn_output)
        # Layer normalization and residual connection
        return self.layernorm2(out1 + ffn_output)

    def compute_output_shape(self, input_shape):
        return input_shape

# Define a custom layer for token and position embedding
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
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
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

