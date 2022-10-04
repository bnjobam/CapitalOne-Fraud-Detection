import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier as abc
import pickle

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

trainX=pd.read_csv('BrianN/trainX.csv')
trainY=pd.read_csv('BrianN/trainY.csv')
testX=pd.read_csv('BrianN/testX.csv')
testY=pd.read_csv('BrianN/testY.csv')

abmodel=abc()
abmodel.fit(trainX,trainY)
Ya=abmodel.predict(testX)
print(abmodel.score(testX,testY))

cma=one_hot_encode(list(testY)).T.dot(one_hot_encode(list(Ya)))
cma=pd.DataFrame(cma, columns=['pred_False','pred_True'], index=['true_False','true_True'])
cma['class_accuracy']=[cma.iloc[i,i]/cma.iloc[i,:].sum() for i in range(cma.shape[0])]

cma.to_csv('BrianN/models/confusion_matrix_abmodel.csv')
classifier=open('BrianN/models/abmodel.pickle','wb')
pickle.dump(abmodel,classifier)
classifier.close()

importance=np.column_stack((np.array(trainX.columns),np.array(abmodel.feature_importances_)))
pd.DataFrame(importance,columns=['variables','importance']).to_csv('BrianN/models/variable_importance_abmodel.csv')
