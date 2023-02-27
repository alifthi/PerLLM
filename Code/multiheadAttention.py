import tensorflow as tf
from dotProdAttention import scaledDotProductAttention 

class multiheadAttention(tf.keras.layers.Layer):
    def __init__(self,Dv,Dk,nHead):
        super().__init__()
        self.Dv = Dv
        self.Dk = Dk
        self.nHead = nHead
    def build(self,inputDim):
        random = tf.random_normal_initializer()
        self.Wo = tf.Variable(initial_value = random(shape = [self.nHead*self.Dv,inputDim[0][-1]],dtype='float'),
                            trainable=True,name = 'query weights')
    def call(self,inputs):
        heads = []
        for i in range(self.nHead):
            heads.append(scaledDotProductAttention(Dv = self.Dv,Dk = self.Dk)(inputs))
        multiHead = tf.concat(heads,axis = -1)
        return tf.matmul(multiHead,self.Wo)