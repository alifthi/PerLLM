from model.model import model
from dataRelated.utils import utils
import numpy as np
dataDir = r"~/Documents/project/PerLLm/Data/Data.json"
util = utils(dataDir)
util.loadData()
decoderInput,decoderOutput,decoderVocabSize = util.preprocess(util.data)

model = model(inputSize = [np.shape(decoderInput)[1],np.shape(decoderInput)[1]],
              decoderVocabSize=decoderVocabSize,encoderVocabSize = decoderVocabSize)

model.net = model.buildModel()
model.compileModel()
model.trainModel([decoderInput,decoderInput,decoderOutput])