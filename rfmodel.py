import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier as rfc
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

rfmodel=rfc(n_estimators=100,n_jobs=-1)
rfmodel.fit(trainX,trainY)
Yr=rfmodel.predict(testX)
print(rfmodel.score(testX,testY))

cmr=one_hot_encode(list(testY)).T.dot(one_hot_encode(list(Yr)))
cmr=pd.DataFrame(cmr, columns=['pred_False','pred_True'], index=['true_False','true_True'])
cmr['class_accuracy']=[cmr.iloc[i,i]/cmr.iloc[i,:].sum() for i in range(cmr.shape[0])]

cmr.to_csv('BrianN/models/confusion_matrix_rfmodel.csv')
classifier=open('BrianN/models/rfmodel.pickle','wb')
pickle.dump(rfmodel,classifier)
classifier.close()

importance=np.column_stack((np.array(trainX.columns),np.array(rfmodel.feature_importances_)))
pd.DataFrame(importance,columns=['variables','importance']).to_csv('BrianN/models/variable_importance_rfmodel.csv')
