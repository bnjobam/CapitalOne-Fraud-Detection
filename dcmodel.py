import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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

dcmodel=DecisionTreeClassifier()
dcmodel.fit(trainX,trainY)
Yl=dcmodel.predict(testX)
print(dcmodel.score(testX,testY))

cml=one_hot_encode(list(testY)).T.dot(one_hot_encode(list(Yl)))
cml=pd.DataFrame(cml, columns=['pred_False','pred_True'], index=['true_False','true_True'])
cml['class_accuracy']=[cml.iloc[i,i]/cml.iloc[i,:].sum() for i in range(cml.shape[0])]

cml.to_csv('BrianN/models/confusion_matrix_dcmodel.csv')
classifier=open('BrianN/models/dcmodel.pickle','wb')
pickle.dump(dcmodel,classifier)
classifier.close()

importance=np.column_stack((np.array(trainX.columns),np.array(dcmodel.feature_importances_)))
pd.DataFrame(importance,columns=['variables','importance']).to_csv('BrianN/models/variable_importance_dcmodel.csv')
