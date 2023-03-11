import tensorflow as tf
# scales dot product attention
class scaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self,Dv,Dk,masked = False) -> None:
        super().__init__()
        self.Dk = Dk
        self.Dv = Dv
        self.masked = masked
    def build(self,inputDim):
        random = tf.random_normal_initializer()
        self.wQ = tf.Variable(initial_value = random(shape = [inputDim[0][-1],self.Dk],dtype='float'),
                            trainable=True,name = 'query weights')
        self.wK = tf.Variable(initial_value = random(shape = [inputDim[1][-1],self.Dk],dtype='float'),
                            trainable=True,name = 'key weights')
        self.wV = tf.Variable(initial_value = random(shape = [inputDim[2][-1],self.Dv],dtype='float'),
                            trainable=True,name = 'value weights')

    def call(self,inputs):
        Q = tf.matmul(inputs[0],self.wQ)
        K = tf.matmul(inputs[1],self.wK)
        V = tf.matmul(inputs[2],self.wV)
        dotProd = tf.matmul(Q,K,transpose_b=True)/64.0
        if  self.masked:
            tensor = tf.ones(dotProd.shape[1:])
            self.mask = -1e10 *tf.linalg.band_part(tensor, 0, -1)
            dotProd += self.mask
        return tf.matmul(tf.keras.activations.softmax(dotProd),V)