# Bring in Custom L1 distance layer module 

# Import dependencies 
import tensorflow as tf
from tensorflow.keras.layers import Layer

# custom L1 dist layer : when we load siamese h5 model we need  to pass through custom objects which is L1 d
class L1Dist(Layer): # custom layer (Layer class is passes)
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # inheritance 

    # core fxn 
    def call(self, inputs):
        # Indexing [0] handles the nested list structure
        input_embedding = inputs[0]
        validation_embedding = inputs[1]
        
        return tf.math.abs(input_embedding - validation_embedding) # calculates similarity 

