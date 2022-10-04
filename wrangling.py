import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm as tq
from datetime import datetime, timedelta

mydata=pd.read_csv('BrianN/capitaldf.csv')

account_age=[ (datetime.strptime(mydata.transactionDateTime[i],'%Y-%m-%dT%H:%M:%S')-datetime.strptime(mydata.accountOpenDate[i],'%Y-%m-%d')).seconds for i in range(mydata.shape[0])]
account_time_to_exp=[(datetime.strptime(mydata.currentExpDate[i],'%m/%Y')-datetime.strptime(mydata.transactionDateTime[i],'%Y-%m-%dT%H:%M:%S')).seconds for i in range(mydata.shape[0])]
time_since_address_change=[ (datetime.strptime(mydata.transactionDateTime[i],'%Y-%m-%dT%H:%M:%S')-datetime.strptime(mydata.dateOfLastAddressChange[i],'%Y-%m-%d')).seconds for i in range(mydata.shape[0])]
transaction_timestamp=[datetime.timestamp(datetime.strptime(dtacapdf.transactionDateTime[i],'%Y-%m-%dT%H:%M:%S')) for i in range(dtacapdf.shape[0])]

dtacapdf.loc[:,'transaction_timestamp']=transaction_timestamp
mydata.loc[:,'account_age']=account_age
mydata.loc[:,'account_time_to_exp']=account_time_to_exp
mydata.loc[:,'time_since_address_change']=time_since_address_change

def get_data(df):
    client=list(set(dtacapdf.accountNumber))
    data=[]
    transaction_cnt=[]
    reverse_cnt=[]
    transaction_dollar=[]
    reverse_dollar=[]
    for i in tq(range(len(client)),desc='Extracting data...'):
        new=df.loc[df.accountNumber==client[i]]
        new=new.sort_values(by='transaction_timestamp')
        new.index=list(range(new.shape[0]))
        cols=list(set([r for r in new.columns if r.find('ime')==-1])-{'currentBalance'})
        #to find unique transactions we need to eliminate columns that will be different for multiple swiping
        unique_index=new.loc[:,cols].drop_duplicates().index
        new=new.iloc[unique_index,:]
        types=list(set(new.transactionType))
        for typ in types:
            newt=new.loc[new.transactionType==typ]
            if typ=='REVERSAL':
                reverse_cnt.append(newt.shape[0])
                reverse_dollar.append(sum(newt.transactionAmount))
            else:
                transaction_cnt.append(newt.shape[0])
                transaction_dollar.append(sum(newt.transactionAmount))
        newd=new.loc[new.transactionType!='REVERSAL']
        newd.index=list(range(newd.shape[0]))
        transaction_rate=[sum(newd.transactionAmount[:i+1])/sum(newd.account_age[:i+1]) for i in range(newd.shape[0])]
        transaction_ave=[sum(newd.transactionAmount[:i+1])/(i+1) for i in range(newd.shape[0])]
        newd.loc[:,'transaction_rate']=transaction_rate
        newd.loc[:,'transaction_ave']=transaction_ave
        for i in tq(range(newd.shape[0]),desc='Saving...'):
            data.append(list(np.array(newd.iloc[i,:])))
    return data,[transaction_cnt,transaction_dollar,reverse_cnt,reverse_dollar]


usefull_data,transaction_summary=get_data(mydata)

usefull_df=pd.DataFrame(usefull_data, columns=list(mydata.columns)+['transaction_rate','transaction_ave'])
usefull_df.to_csv('BrianN/fraud_detection.csv')

(fig, ax) = plt.subplots(2, 2, figsize=(18, 8))
titles=['transaction_cnt','transaction_amount','reverse_cnt','reverse_amount']
for i in range(4):
    ax[i//2,i%2].set_title(titles[i])
    ax[i//2,i%2].set_xlabel("bins")
    ax[i//2,i%2].set_ylabel('frequency')
    ax[i//2,i%2].hist(transaction_summary[i], bins='auto')


plt.figure(figsize=(12,20))
plt.show()
fig.savefig('BrianN/plots/Histogram_transactionAmount.png')
plt.close(fig)
