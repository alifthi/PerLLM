import tensorflow as tf
from Encoder import Encoder as encoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numDecoder = 6,Dv = 64,Dk = 512,nHead = 8):
        super().__init__()
        self.numDecoder = numDecoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim
        self.encoder = encoder()
