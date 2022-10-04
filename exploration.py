import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from zipfile import ZipFile
import urllib.request, json
from tqdm import tqdm as tq
import scipy
#import scipy.stat

os.mkdir('BrianN')
os.mkdir('BrianN/plots')

def plots(df,title):
    df.index=list(range(df.shape[0]))
    if isinstance(df[0],(str,np.bool_)):
        names=list(set(df))
        freq=[names,[0 for i in range(len(names))]]
        for j in df:
            freq[1][names.index(j)]+=1
        if len(names)>19: #it will be hard to look into the labels and bars sor a plot is convenient
            fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(20,16))
            ax.plot(freq[1])
            ax.set_ylabel('frquency')
            ax.set_xlabel(f'{names[0]}...{names[-1]}_labels')
            ax.set_title(title)
            # plt.show()
            fig.savefig(f'BrianN/plots/{title}.png')
            plt.close(fig)
        else:
            fig,ax=plt.subplots(figsize=(20,16))
            ax.bar(freq[0],height=freq[1], tick_label=names)
            ax.set_ylabel('frquency')
            ax.set_xlabel('labels')
            ax.set_title(title)
            # plt.show()
            fig.savefig(f'BrianN/plots/{title}.png')
            plt.close(fig)
    else:
        fig,ax=plt.subplots()
        df.plot.hist(bins=7)
        ax.set_ylabel('frquency')
        ax.set_xlabel('bins')
        ax.set_title(title)
        # plt.show()
        fig.savefig(f'BrianN/plots/{title}.png')
        plt.close(fig)
    return

#copied from http://www.insightsbot.com/fitting-probability-distributions-with-python/
class Distribution(object):

    def __init__(self,dist_names_list = []):
        self.dist_names = ['norm','lognorm','expon','pareto','chi2','cauchy','exponnorm', 'gamma', 'laplace','loglaplace','triang']
        self.dist_results = []
        self.params = {}

        self.DistributionName = ""
        self.PValue = 0
        self.Param = None

        self.isFitted = False
    def Fit(self, y):
        self.dist_results = []
        self.params = {}
        for dist_name in self.dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)

            self.params[dist_name] = param
            #Applying the Kolmogorov-Smirnov test
            D, p = scipy.stats.kstest(y, dist_name, args=param);
            self.dist_results.append((dist_name,p))

        #select the best fitted distribution
        sel_dist,p = (max(self.dist_results,key=lambda item:item[1]))
        #store the name of the best fit and its p value
        self.DistributionName = sel_dist
        self.PValue = p

        self.isFitted = True
        return self.DistributionName,self.PValue


urllib.request.urlretrieve("https://github.com/CapitalOneRecruiting/DS/raw/master/transactions.zip","BrianN/transactions.zip")
with ZipFile("BrianN/transactions.zip",'r') as zips:
    data=zips.extract('BrianN/transactions.txt')
with open(data, 'r') as f:
    mydata=f.readlines()

capitaldict1=[]
for i in tq(range(len(mydata)), desc='Extracting'):
    ss=[]
    for h in json.loads(mydata[1]).keys():#I used 1 because the order changes in other enteries
        ff=json.loads(mydata[i])[h]
        if isinstance(ff,(float, bool)):
            ss.append(ff)
        elif len(ff)>0:

            try:
                ss.append(int(ff))
            except:
                ss.append(ff)
        else:
            ss.append(None)
    capitaldict1.append(ss)



capitaldf=pd.DataFrame(capitaldict1, columns=list(json.loads(mydata[1]).keys()))

datacap=capitaldf.dropna(axis=1)
#this is to drop all variables with a missing value so we can easily inspect them
#capitaldf.loc[:,list(set(capitaldf.columns)-set(datacap.columns))]
#From here we saw that ['echoBuffer','merchantState','merchantCity','merchantZip','posOnPremises','recurringAuthInd']
#have only NAs
capitaldf.loc[:,['echoBuffer','merchantState','merchantCity','merchantZip','posOnPremises','recurringAuthInd']].describe().to_csv('NA_variables.csv')
datacapdf=capitaldf.drop(['echoBuffer','merchantState','merchantCity','merchantZip','posOnPremises','recurringAuthInd'],axis=1)
#Summary statistics for other NA_variables

num_col=[]
cat_col=[]
for col in datacapdf.columns:
    if isinstance(datacapdf.loc[0,col],(str,np.bool_)):
        cat_col.append(col)
    else:
        num_col.append(col)


cat_col_summary=capitaldf.loc[:,cat_col].describe()
null_values_cnt=np.array([capitaldf.shape[0]-cat_col_summary.iloc[0,i] for i in range(cat_col_summary.shape[1])])
unique_values_less_20=np.array([list(set(capitaldf.loc[:,col])) if cat_col_summary.loc['unique',col]<20 else [] for col in cat_col_summary.columns ])
cat_col_summary.loc['null_value_cnt',:]=null_values_cnt
cat_col_summary.loc['unique_values',:]=unique_values_less_20
cat_col_summary.to_csv('BrianN/category_variables_summary.csv')

num_col_summary=capitaldf.loc[:,num_col].describe()
null_values_cntn=np.array([capitaldf.shape[0]-num_col_summary.iloc[0,i] for i in range(num_col_summary.shape[1])])
num_col_summary.loc['null_value_cnt',:]=null_values_cntn
num_col_summary.to_csv('BrianN/numerical_valriables_summary.csv')

datacapdf1=datacapdf.dropna(axis=0)
k=20
fig, ax=plt.subplots(ncols=1,nrows=1, figsize=(16,11))
ax.hist(dtacapdf.transactionAmount, bins=np.linspace(min(dtacapdf.transactionAmount),max(dtacapdf.transactionAmount),k),color='green',label='all')
ax.hist(dtacapdf.loc[dtacapdf.isFraud!=True].transactionAmount, bins=np.linspace(min(dtacapdf.transactionAmount),max(dtacapdf.transactionAmount),k),color='yellow',label='False')
ax.hist(dtacapdf.loc[dtacapdf.isFraud==True].transactionAmount, bins=np.linspace(min(dtacapdf.transactionAmount),max(dtacapdf.transactionAmount),k),color='red',label='True')
ax.set_xlabel('classes')
ax.set_ylabel('Frequency')
ax.set_title('Histogram_transactionAmount')
ax.legend()
plt.show()
fig.savefig('BrianN/plots/Compact_Histogram_transactionAmount.png')
plt.close(fig)

test_dist=Distribution()
dd=np.histogram(datacapdf1.loc[datacapdf1.isFraud==True].transactionAmount,np.linspace(min(datacapdf1.transactionAmount),max(datacapdf1.transactionAmount),k))
dd1=np.histogram(datacapdf1.loc[datacapdf1.isFraud!=True].transactionAmount,np.linspace(min(datacapdf1.transactionAmount),max(datacapdf1.transactionAmount),k))
dd2=np.histogram(datacapdf1.transactionAmount,np.linspace(min(datacapdf1.transactionAmount),max(datacapdf1.transactionAmount),k))

print(test_dist.Fit(dd2[0]))
#the whole data for transactionAmount follows a loglaplace distribution, same as the isFraud=False
#but isFraud=True follows a pareto distribution (roughly meaning 80% of the data is in 20% of the classes)

for col in tq(['creditLimit', 'availableMoney',
       'transactionAmount', 'merchantName',
       'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode',
       'merchantCategoryCode', 'cardPresent',
       'cardCVV', 'enteredCVV',
       'cardLast4Digits', 'transactionType', 'currentBalance',
       'expirationDateKeyInMatch']):
    plots(datacapdf1.loc[:,col],f'{col}_distribution')
    plots(datacapdf1.loc[datacapdf1.isFraud==True,col],f'{col}_distribution_True')
    plots(datacapdf1.loc[datacapdf1.isFraud!=True].loc[:,col],f'{col}_distribution_False')

datacapdf1.to_csv('BrianN/capitaldf.csv')
