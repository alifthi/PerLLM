import tensorflow as tf
from tensorflow.keras import layers as ksl
class Encoder(tf.keras.layers.Layer):
    def __init__(self,denseDim = 512,Dv = 64,Dk = 256,nHead = 8):
        super().__init__()
        self.Dk = Dk                        
        self.Dv = Dv
        self.nHead = nHead                  # number of heads in multihead attention
        self.denseDim = denseDim            # dimention of middle feed forward network    
    def build(self,inputShape):
        random = tf.random_normal_initializer()
        self.Wo = tf.Variable(initial_value = random(shape = [self.nHead*self.Dv,inputShape[-1]],dtype='float'),
                            trainable=True,name = 'query weights')
        self.wQ = tf.Variable(initial_value = random(shape = [inputShape[-1],self.Dk],dtype='float'),
                            trainable=True,name = 'query weights')
        self.wK = tf.Variable(initial_value = random(shape = [inputShape[-1],self.Dk],dtype='float'),
                            trainable=True,name = 'key weights')
        self.wV = tf.Variable(initial_value = random(shape = [inputShape[-1],self.Dv],dtype='float'),
                            trainable=True,name = 'value weights')
        self.layerNormalization1 = ksl.LayerNormalization()
        self.dense1 = ksl.Dense(self.denseDim,activation = 'relu')
        self.dense2 = ksl.Dense(inputShape[-1])
        self.layerNormalization2 = ksl.LayerNormalization()
    def call(self,inputs):
        y = inputs
        heads = []
        for _ in range(self.nHead):
            Q = tf.matmul(y,self.wQ)
            K = tf.matmul(y,self.wK)
            V = tf.matmul(y,self.wV)
            dotProd = tf.matmul(Q,K,transpose_b=True)/tf.cast(64,dtype='float32')
            heads.append(tf.matmul(tf.keras.activations.softmax(dotProd),V))
        multiHead = tf.concat(heads,axis = -1)
        multiHead = tf.matmul(multiHead,self.Wo)
        x =  ksl.Add()([y,multiHead])
        x =  self.layerNormalization1(x)
        ff = self.dense1(x)
        ff = self.dense2(ff)
        x =  ksl.Add()([ff,x])
        y =  self.layerNormalization2(x)
        return y
