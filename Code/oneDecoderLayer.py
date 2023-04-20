import tensorflow as tf
from Encoder import Encoder as encoder
from tensorflow.keras import layers as ksl
class Decoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 2048,numDecoder = 6,Dv = 64,Dk = 256,nHead = 8):
        super().__init__()
        self.numDecoder = numDecoder        # number of encoders
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim
    def build(self,inputShape):
        random = tf.random_normal_initializer()
        self.Wo = tf.Variable(initial_value = random(shape = [self.nHead*self.Dv,inputShape[0][-1]],dtype='float'),
                            trainable=True,name = 'query weights')
        self.wQ = tf.Variable(initial_value = random(shape = [inputShape[0][-1],self.Dk],dtype='float'),
                            trainable=True,name = 'query weights')
        self.wK = tf.Variable(initial_value = random(shape = [inputShape[0][-1],self.Dk],dtype='float'),
                            trainable=True,name = 'key weights')
        self.wV = tf.Variable(initial_value = random(shape = [inputShape[0][-1],self.Dv],dtype='float'),
                            trainable=True,name = 'value weights')
        self.layerNormalization1 =  ksl.LayerNormalization()
        self.layerNormalization2 = ksl.LayerNormalization()
        self.layerNormalization3 = ksl.LayerNormalization()
        self.dense1 = ksl.Dense(self.denseDim,activation = 'relu')
        self.dense2 = ksl.Dense(inputShape[-1])
    def call(self,inputs):
        y = inputs[0] + self.posEncoding
        heads = []
        for _ in range(self.nHead):
            Q = tf.matmul(y,self.wQ)
            K = tf.matmul(y,self.wK)
            V = tf.matmul(y,self.wV)
            dotProd = tf.matmul(Q,K,transpose_b=True)/tf.cast(64,dtype='float32')
            tensor = tf.ones(dotProd.shape[1:])
            self.mask = -1e10 *tf.linalg.band_part(tensor, 0, -1)
            dotProd += self.mask
            heads.append(tf.matmul(tf.keras.activations.softmax(dotProd),V))
        multiHead = tf.concat(heads,axis = -1)
        multiHead = tf.matmul(multiHead,self.Wo)
        x =  ksl.Add()([y,multiHead])
        x = self.layerNormalization1(x)
        encoderOut = inputs[1]
        heads = []
        for _ in range(self.nHead):
            Q = tf.matmul(x,self.wQ)
            K = tf.matmul(encoderOut,self.wK)
            V = tf.matmul(encoderOut,self.wV)
            dotProd = tf.matmul(Q,K,transpose_b=True)/tf.cast(64,dtype='float32')
            heads.append(tf.matmul(tf.keras.activations.softmax(dotProd),V))
        multiHead = tf.concat(heads,axis = -1)
        attention = tf.matmul(multiHead,self.Wo)        
        x =  ksl.Add()([attention,x])
        x =  self.layerNormalization2(x)
        ff = self.dense1(x)
        ff = self.dense2(ff)
        x =  ksl.Add()([ff,x])
        y =  self.layerNormalization3(x)
        return y