from model import model
from utils import utils
import numpy as np
dataDir = r"C:\Users\alifa\Documents\AI\DATA\NLP\cnn_dailymail\\"
util = utils(dataDir)
util.loadData()
text,decoderInput,decoderOutput,decoderVocabSize,encoderVocabSize = util.preprocess(util.trainData)

print(np.shape(text))
print(np.shape(decoderInput))
print(decoderVocabSize)
# model = model(inputSize = [np.shape(text)[1],np.shape(decoderInput)[1]],decoderVocabSize=decoderVocabSize)
model = model(inputSize = [np.shape(text)[1],np.shape(decoderInput)[1]],decoderVocabSize=decoderVocabSize,encoderVocabSize = encoderVocabSize)

model.net = model.buildModel()
model.compileModel()
model.trainModel([text,decoderInput,decoderOutput])