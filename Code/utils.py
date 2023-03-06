import tensorflow as tf
import pandas as pd
import numpy as np

class utils:
    def __init__(self,dataDir):
        self.dataDir = dataDir
    def loadData(self):
        self.trainData = pd.read_csv(self.dataDir + 'train.csv')
        self.testData = pd.read_csv(self.dataDir + 'test.csv')
    @staticmethod
    def preprocess(data):
        data['article'] = data['article'].str.replace('[^\w\s]','')
        data['article'] = data['article'].str.replace('\([^)]*\)','')
