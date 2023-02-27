import tensorflow as tf
from dotProdAttention import scaledDotProductAttention 

class multiheadAttention(tf.keras.layers.Layer):
    def __init__(self,Dv,Dk,nHead):
        super().__init__()
        self.Dv = Dv
        self.Dk = Dk
        self.nHead = nHead