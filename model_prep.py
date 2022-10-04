import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm as tq
from sklearn .model_selection import train_test_split
import os



usefull_df=pd.read_csv('BrianN/fraud_detection.csv')
usefull_df.index=list(range(usefull_df.shape[0]))
datadf=pd.get_dummies(usefull_df.drop(['accountNumber','customerId','transactionDateTime','merchantName','dateOfLastAddressChange','accountOpenDate','currentExpDate','transaction_timestamp','isFraud'], axis=1))
target=usefull_df.isFraud*1
#['accountNumber','customerId','transactionDateTime','merchantName','dateOfLastAddressChange','accountOpenDate','currentExpDate','isFraud']
#these variables are either useless in analysis like accountNumber or create useless variations like dates, we can see that from the plots
#merchantName is vast and will have a large number of dummies, we could group them according to frequency of occurence

#this is to ballance the siFraud so as not to have high accuracy with only one class classification
true_index=usefull_df.loc[usefull_df.isFraud==True].index
n=28 #to get as much isFraud==False data as possible
indexes=list(range(n*len(true_index)))+n*list(true_index)
our_data=datadf.iloc[indexes,:]
our_target=target[indexes]

trainX,testX,trainY,testY=model_selection.train_test_split(our_data,our_target,test_size=0.15, random_state=1)

trainX.to_csv('BrianN/trainX.csv')
trainY.to_csv('BrianN/trainY.csv')
testX.to_csv('BrianN/testX.csv')
testY.to_csv('BrianN/testY.csv')

os.mkdir('BrianN/models')
