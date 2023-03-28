from model import model
from utils import utils
import numpy as np
dataDir = r"C:\Users\alifa\Documents\AI\DATA\NLP\cnn_dailymail\\"
util = utils(dataDir)
util.loadData()
text,summary,decoderVocabSize = util.preprocess(util.trainData)
print(np.shape(text))
model = model(inputSize = [np.shape(text)[1]],decoderVocabSize=decoderVocabSize)
model.net = model.buildModel()
model.compileModel()
model.trainModel([text,summary,decoderVocabSize])