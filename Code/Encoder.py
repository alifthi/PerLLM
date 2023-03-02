import tensorflow as tf
from multiheadAttention import multiheadAttention 
from tensorflow.keras import layers as ksl
from tensorflow.keras import layers as ksl
class Encoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numEncoder = 6,Dv = 64,Dk = 512,nHead = 8):
        super().__init__()
        self.numEncoder = numEncoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim            # dimention of middle feed forward network    
    def build(self,inputShape):
        self.attention = multiheadAttention(Dk = self.Dk,Dv = self.Dv,nHead=self.nHead)