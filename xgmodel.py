import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost
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
model=xgboost.XGBClassifier(n_jobs=-1,learning_rate=0.2, verbosity=1)
xgmodel=model.fit(trainX,trainY)

Y=xgmodel.predict(testX)
xgmodel.score(testX,testY)

cm=one_hot_encode(list(testY)).T.dot(one_hot_encode(Y))
cm=pd.DataFrame(cm, columns=['pred_False','pred_True'], index=['true_False','true_True'])
cm['class_accuracy']=[cm.iloc[i,i]/cm.iloc[i,:].sum() for i in range(cm.shape[0])]
cm.to_csv('BrianN/models/confusion_matrix_xgmodel.csv')
classifier=open('BrianN/models/xgmodel.pickle','wb')
pickle.dump(xgmodel,classifier)
classifier.close()

importance=np.column_stack((np.array(trainX.columns),np.array(xgmodel.feature_importances_)))
pd.DataFrame(importance,columns=['variables','importance']).to_csv('BrianN/models/variable_importance_xgmodel.csv')
