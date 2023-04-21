from model import model
from utils import utils
import numpy as np
dataDir = r"C:\Users\alifa\Documents\AI\DATA\NLP\cnn_dailymail\\"
util = utils(dataDir)
util.loadData()
text,decoderInput,decoderOutput,decoderVocabSize,encoderVocabSize = util.preprocess(util.trainData)

model = model(inputSize = [np.shape(text)[1],np.shape(decoderInput)[1]],decoderVocabSize=decoderVocabSize,encoderVocabSize = encoderVocabSize)

model.net = model.buildModel()
model.compileModel()
model.trainModel([text,decoderInput,decoderOutput])