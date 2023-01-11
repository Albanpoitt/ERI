# -*- coding: utf-8 -*-
# file directory:  C:/EDHEC/Cours/S2/S2_b/ERI/Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.linear_model import LinearRegression
#from datetime import datetime


####################################################################################


#Import of data


#Import Monthly prices
prices = pd.read_csv("Prices.csv", sep = ',', parse_dates=['Name'],encoding='utf-8-sig')
prices.rename(columns={'Name':'dates'}, inplace = True)
prices = prices.groupby(pd.Grouper(key='dates',freq='M')).max()
#prices.columns = data_2010.index #get the same name for the stocks

#Import Total return index
TRI_original = pd.read_csv("Total Return Index.csv", sep = ',', parse_dates=['Name'],encoding='utf-8-sig',low_memory=False)
TRI_original.rename(columns={'Name':'dates'}, inplace = True)
TRI_original = TRI_original.groupby(pd.Grouper(key='dates',freq='M')).max()
TRI_original.drop(TRI_original.tail(1).index,inplace=True)#Erase 2020 data

#Import Monthly Market Cap
Cap = pd.read_csv("Market Capitalization.csv", sep = ',', parse_dates=['Name'],encoding='utf-8-sig')
Cap.rename(columns={'Name':'dates'}, inplace = True)
Cap = Cap.groupby(pd.Grouper(key='dates',freq='M')).max()
Cap.drop(Cap.tail(1).index,inplace=True)

#Import MTBV
BV= pd.read_csv("BV.csv", sep = ',',encoding='utf-8-sig',parse_dates=['dates'])
BV = BV.groupby(pd.Grouper(key='dates',freq='M')).max()
BV.drop(BV.head(2).index,inplace=True)
BV.drop(BV.tail(1).index,inplace=True)

#Import Monthly Rf
RF = pd.read_csv("Europe_RF.csv", sep = ',', parse_dates=['Name'],encoding='utf-8-sig')
RF.rename(columns={'Name':'dates'}, inplace = True)
RF = RF.groupby(pd.Grouper(key='dates',freq='M')).max()
RF.drop(RF.head(4).index,inplace=True)
RF.drop(RF.tail(1).index,inplace=True)

#Import Monthly Market Returns
Mkt_original = pd.read_csv("Market.csv", sep = ',',parse_dates=['dates'],encoding='utf-8-sig')
Mkt_original = Mkt_original.groupby(pd.Grouper(key='dates',freq='M')).max()
Mkt_original.drop(Mkt_original.head(2).index,inplace=True)
Mkt_original.drop(Mkt_original.tail(1).index,inplace=True)

#Import and convert the characterstic tables to an array
period = np.array(range(2008,2020))

for i in period:
    x = str(i)+".csv"
    globals()["data_" + str(i)] = pd.read_csv(x, index_col='#NAME?')

#Store the characterisitics in variable data_date
charac = pd.read_csv("List of Characteristics.csv", names=['Characteristics'])
charac = charac.Characteristics.to_list()
charac_add_Cap = charac.copy()
charac_add_Cap.append('Market Cap')
charac_add_Cap.append('BTMV')

charac_w_constant = charac_add_Cap.copy()
charac_w_constant.append('Constant')

#Import ESG ETF
ESG= pd.read_excel('ESG.xlsx', index_col='dates',parse_dates=['dates'])
ESG = ESG.groupby(pd.Grouper(level='dates',freq='M')).max()


#################################################################################################



#Functions Creation

#Get the stocks that verify the characteristics 

#Function to get the stocks that verify a particular characteristics
# def stocks(characterisitics, data_date):
#     x=[]
#     for i in range(600):
#         if pd.isna(data_date.loc[:,characterisitics].iloc[i])==False:
#             x.append(data_date.index[i])
#     return x

# #Stock that verify all characteristics for a given year
def test(characteristics,data_date):
    x=[]
    n = len(characteristics)
    for i in range(600): #loop for the stocks
        m=0 #count 
        for k in characteristics: #loop to check if all charac are verified
            if pd.isna(data_date.loc[:,k].iloc[i])==False:
                m=m+1
        if m==n:
            x.append(data_date.index[i])
    return x

#Function that return the type of the characteristic
def type_char(characterisitics):
    data=data_2019.copy()
    data['Market Cap']=Cap.loc['2019-12-31',:]
    data['BTMV']=BV.loc['2019-12-31',:]
    i=0
    while pd.isna(data.loc[:,characterisitics][i])==True:
          i=i+1
    if data.loc[:,characterisitics][i] =='Y' or data.loc[:,characterisitics][i] == 'N':
        return ('Binary')
    if pd.api.types.is_number(data.loc[:,characterisitics][i]):
        return ('Numerical')
    else:
        return ('Qualitative')
    
#Past version of the function weight
# def weight(data_date,stocks):
#     w = pd.DataFrame(columns=charac, index=stocks) #Creation of the dataframe for the weight
#     n = len(stocks)
#     for k in charac:
#         if type_char(k)=='Binary':
#             Y=0
#             N=0
    
#             for i in stocks: #Loop for stocks
#                 if data_date.loc[i,k]=='Y':
#                     Y=1+Y
#                 else:
#                     N=N+1
#             for i in stocks: #Loop for stocks
#                 if data_date.loc[i,k]=='Y':
#                     w.loc[i,k]=1/n
#                 else:
#                     w.loc[i,k]=-1/n*(Y/N)
#         if type_char(k)=='Numerical':
#             p=data_date.loc[stocks,k].rank(numeric_only=True)/(len(stocks)+1)-0.5
#             for i in stocks:
#                 w[k]=p
#         if type_char(k)=='Qualitative':
#             w[k]=1/len(stocks)
#     return w




# #Weighted average for a specific characteristic for a specific year (Not used at the moment)
# def portfolio_prices_Year(date, data_date):
#     period = prices.index[(prices.index.year==date)]#dates for the year
#     portfolio = pd.DataFrame(columns=charac, index=period)
#     u = test(charac,data_date) #all stocks for the year
#     portfolio.insert(len(portfolio.columns),"Nb of Stocks",True)
#     #Get the stocks that provide data for the month
#     for t in period:#loop on the month of the year
#         new_u = [] # New list of stocks
#         for s in u: #loop on the stocks to get the stocks reporting data
#             if pd.isna(prices.loc[t,s])==False:
#                 new_u.append(s)
#         #compute the weight for the month
#         w = weight(data_date,new_u)
#         for c in charac:
#             sum_price = 0
#             for s in new_u:
#                 sum_price = sum_price + prices.loc[t,s]*w.loc[s,c]
#             portfolio.loc[t,c]=sum_price
#             portfolio.loc[t,'Nb of Stocks']=len(new_u)
#             #print(sum_price)
#     return portfolio

# #Weight for all dates (Not used at the moment)
# def portfolio_prices_All(Start_Year, End_Year):
#     i = 0
#     for date in range (Start_Year, End_Year+1,1):
#          data_date = globals()["data_" + str(date)]
#          if i==0:
#             print(i)
#             portfolio = portfolio_prices_Year(date,data_date)
#             i+=1
#          else: 
#             print(i)
#             a = portfolio_prices_Year(date,data_date)
#             portfolio = pd.concat([portfolio, a])
#     return portfolio


#Function to compute the Historic Total Returns for every Stock 
def returns(price):
    df_returns=pd.DataFrame(columns=price.columns,index=price.index)
    i=0
    while i<len(price.columns): #Loop on the number of stocks
        j=0
        while j<len(price.iloc[:,0]): #Loop on the number of dates
            if j==0:
                df_returns.iloc[j,i]=np.nan
            else:
                if pd.isna(price.iloc[j,i])==True or pd.isna(price.iloc[j-1,i])==True:
                    df_returns.iloc[j,i]=np.nan
                else:
                    df_returns.iloc[j,i]=np.log(price.iloc[j,i]/price.iloc[j-1,i])    
            j=j+1
        i=i+1
    df_returns =df_returns.astype(float)
    return df_returns

#For a given date we compute the weight for each characterisitc
def weight(data_date,stocks):
    w = pd.DataFrame(columns=charac_add_Cap, index=stocks) #Creation of the dataframe for the weight
    data=data_date.copy()
    data=data.loc[stocks].copy()
    for k in charac_add_Cap:
        if type_char(k)=='Binary':
            Y=0
            N=0
            for i in stocks: #Loop for stocks
                if data.loc[i,k]=='Y':
                    Y=1+Y
                else:
                    N=N+1
            for i in stocks: #Loop for stocks
                if data.loc[i,k]=='Y':
                    w.loc[i,k]=1/Y
                else:
                    w.loc[i,k]=-1/N
        if type_char(k)=='Numerical':
            p=data.loc[stocks,k].rank(numeric_only=True)/(len(stocks)+1)-0.5
            #rescaling of the weight 
            pos = sum(x for x in p if x > 0)
            neg = sum(x for x in p if x < 0)
            for i in range(len(p)):
                if p.iloc[i] > 0:
                    p.iloc[i] = p.iloc[i] / pos
                else:
                    p.iloc[i] = p.iloc[i] / abs(neg)
            w[k]=p
    # p=Cap.loc[period,stocks].rank(numeric_only=True)/(len(stocks)+1)-0.5
    # w['Market Cap']=p
    # q=BV.loc[period,stocks].rank(numeric_only=True)/(len(stocks)+1)-0.5
    # w['BTMV']=q         
    return w

#Function to form the Portfolios' Time Series of Returns
def Port(Returns):
    period = Returns.index
    portfolio = pd.DataFrame(columns=charac_add_Cap, index=period)
    portfolio.insert(len(portfolio.columns),"Constant",True)
    portfolio.insert(len(portfolio.columns),"Nb of Stocks",True)
    #Get the stocks that provide data for the month
    for t in period:#loop on the month
        date=t.year
        data_date = globals()["data_" + str(date)]
        data_date['Market Cap']=Cap.loc[t,:]
        data_date['BTMV']=BV.loc[t,:]
        u = test(charac_add_Cap,data_date)
        new_u = [] # New list of stocks
        for s in u: #loop on the stocks to get the stocks reporting data
            if pd.isna(Returns.loc[t,s])==False and pd.isna(Cap.loc[t,s])==False and pd.isna(BV.loc[t,s])==False:  
                new_u.append(s)
        #compute the weight for the month
        w = weight(data_date,new_u)
        for c in charac_add_Cap:
            sum_returns = 0
            for s in new_u:
                sum_returns = sum_returns + Returns.loc[t,s]*w.loc[s,c]
            portfolio.loc[t,c]=sum_returns 
        sum_constant = 0
        for s in new_u:
            sum_constant = sum_constant + Returns.loc[t,s]*(1/len(new_u)) #Constant is a long only portfolio
        portfolio.loc[t,'Constant'] =sum_constant
        portfolio.loc[t,'Nb of Stocks']=len(new_u)
    portfolio =portfolio.astype(float)
    return portfolio

#Construction of Port with the lag
def Port_lag(Returns):
    period = Returns.index
    portfolio = pd.DataFrame(columns=charac_add_Cap, index=period)
    portfolio.insert(len(portfolio.columns),"Constant",True)
    portfolio.insert(len(portfolio.columns),"Nb of Stocks",True)
    #Get the stocks that provide data for the month
    t_index = 0
    for t in period[:-1]:#loop on the month
        date=t.year
        data_date = globals()["data_" + str(date)]
        data_date['Market Cap']=Cap.loc[t,:]
        data_date['BTMV']=BV.loc[t,:]
        u = test(charac_add_Cap,data_date)
        new_u = [] # New list of stocks
        for s in u: #loop on the stocks to get the stocks reporting data
            if pd.isna(Returns.loc[period[t_index+1],s])==False and pd.isna(Cap.loc[period[t_index+1],s])==False and pd.isna(BV.loc[period[t_index+1],s])==False:  
                new_u.append(s)
        #compute the weight for the month
        w = weight(data_date,new_u)
        for c in charac_add_Cap:
            sum_returns = 0
            for s in new_u:
                if pd.isna(Returns.loc[period[t_index+1],s]) == False: #See if in t+1 you have data to avoid NaN that will mess the code 
                    sum_returns = sum_returns + Returns.loc[period[t_index+1],s]*w.loc[s,c]
            portfolio.loc[period[t_index+1],c]=sum_returns 
        sum_constant = 0
        for s in new_u:
            sum_constant = sum_constant + Returns.loc[period[t_index+1],s]*(1/len(new_u))
        portfolio.loc[period[t_index+1],'Constant'] =sum_constant
        portfolio.loc[period[t_index+1],'Nb of Stocks']=len(new_u)
        t_index+=1
    portfolio = portfolio.astype(float)
    return portfolio

#Function to graph of the evolution of the returns of the portfolio for one characteristic
def graph_prices(Time_Series,Characteristic, rgb):
    df =  Time_Series.reset_index()
    df.plot(x='dates', y=Characteristic, color= rgb)
    plt.show()


#Weigthing proof
# date=2017
# data_date = globals()["data_" + str(date)]
# data_date['Market Cap']=Cap.loc['2017-02-28',:]
# data_date['BTMV']=BV.loc['2017-02-28',:]
# u = test(charac_add_Cap,data_date)
# new_u = [] 
# for s in u: #loop on the stocks to get the stocks reporting data
#     if pd.isna(TRI.loc['2017-02-28',s])==False and pd.isna(Cap.loc['2017-02-28',s])==False and pd.isna(BV.loc['2017-02-28',s])==False:  
#         new_u.append(s)
# w = weight(data_date,new_u,'2017-02-28')
# #w.to_csv(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\w.csv',index=True,header=True)
# TRI.to_csv(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\TRI.csv',index=True,header=True)
# Mkt.to_csv(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Mkt.csv',index=True,header=True)



######################################################################################################



#Analysis


#Compute the Total Returns for the stocks
Mkt=returns(Mkt_original)
Mkt.drop(Mkt.head(1).index,inplace=True)
TRI=returns(TRI_original)
TRI.drop(TRI.head(1).index,inplace=True)
#Portfolios=Port(TRI)

# #Export Portfolios
#Portfolios.to_csv(r'C:\EDHEC\Cours\S2\S2_b\ERI\Python\Returns_from_python.csv',index=True,header=True)

# #Import Portfolios
Portfolios = pd.read_csv('Returns_from_python.csv',index_col=(0),parse_dates=['dates'])

# #Create graph for number of stock and some returns
# NbStocks=Portfolios['Nb of Stocks'].copy()
# del Portfolios['Nb of Stocks']
# graph_prices(NbStocks,'Nb of Stocks') 
# graph_prices(Portfolios,charac[1:4],'b')   
# graph_prices(Portfolios,charac[58:63],'b')  

#Lag adjustments
Mkt_lag=Mkt.copy()
Mkt_lag.drop(Mkt_lag.head(1).index,inplace=True)
TRI_lag = TRI.copy()
TRI_lag.drop(TRI_lag.head(1).index,inplace=True)
#Portfolios_lag=Port_lag(TRI_lag)
#Portfolios_lag.drop(Portfolios_lag.head(1).index,inplace=True)

#Export Portfolios_lag
#Portfolios_lag.to_csv(r'C:\EDHEC\Cours\S2\S2_b\ERI\Python\Returns_from_python_lag.csv',index=True,header=True)

#Import Portfolios_lag
Portfolios_lag = pd.read_csv('Returns_from_python_lag.csv',index_col=(0),parse_dates=['dates'])


################_PERIOD 2008 - 2019_##############

#Computation of the Correlations Table of Returns for 2008 - 2019
Corr=Portfolios.corr()
Corr.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Correlations.xlsx',index=True,header=True)

#Sharpe ratio
#RF = RF.reset_index(drop=True)

Sharpe_ratio = pd.DataFrame(columns=charac_w_constant, index=['Sharpe ratio'])
Excess_return = Portfolios.copy()
for i in charac_w_constant:
    Excess_return.loc[:,i] = Portfolios.loc[:,i] - RF.loc[:,'RF']/(100*12) 
var = Portfolios.var()

for i in charac_w_constant:
    index = charac_w_constant.index(i)
    Sharpe_ratio.loc[:,i]= (12**0.5)*(Excess_return.loc[:,i].mean()/var[index]**0.5)

#Computation of beta
Beta =pd.DataFrame(columns=charac_w_constant, index=['Beta'])

for c in charac_w_constant:
    Cov_matrix=np.cov(Portfolios.loc[:,c],Mkt.loc[:,'Market'])
    Beta.loc['Beta',c]=(Cov_matrix[0][1])/Cov_matrix[1][1]


#Computation of Time Series Statistics
result = scipy.stats.describe(Portfolios.iloc[:,1], ddof=1, bias=False)
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=charac_w_constant)
h=-1
for k in charac_w_constant:
    h=h+1
    l=-1
    v=1
    result = scipy.stats.describe(Portfolios.iloc[:,h], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,h]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,h]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,h]=result[l]
                else:
                    Statistics.iloc[l+1,h]=result[l]

Statistics=Statistics.append(Sharpe_ratio)
Statistics=Statistics.append(Beta)
#Export Statistics
Statistics.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_before_2010.xlsx',index=True,header=True)
#Import Statistics
# Statistics = pd.read_excel(r'Statistics_before_2010.xlsx')

############################## WITH LAG : 2008 - 2019 #######################################

#Corr check
Corr=Portfolios_lag.corr()
Corr.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Correlations_lag.xlsx',index=True,header=True)

#Computation of Sharpe ratio
Sharpe_ratio = pd.DataFrame(columns=charac_w_constant, index=['Sharpe ratio'])
Excess_return = Portfolios_lag.copy()
for i in charac_w_constant:
    Excess_return.loc[:,i] = Portfolios_lag.loc[:,i] - RF.loc[RF.index[1:],'RF']/(100*12) 
var = Portfolios_lag.var()

for i in charac_w_constant:
    index = charac_w_constant.index(i)
    Sharpe_ratio.loc[:,i]= (12**0.5)*(Excess_return.loc[:,i].mean()/var[index]**0.5)
    

#Computation of beta
Beta =pd.DataFrame(columns=charac_w_constant, index=['Beta'])

for c in charac_w_constant:
    Cov_matrix=np.cov(Portfolios_lag.loc[:,c],Mkt_lag.loc[:,'Market'])
    Beta.loc['Beta',c]=(Cov_matrix[0][1])/Cov_matrix[1][1]

#Computation of Time Series Statistics
result = scipy.stats.describe(Portfolios.iloc[:,1], ddof=1, bias=False)
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics_w_lag=pd.DataFrame(index=Stats,columns=charac_w_constant)
h=-1
for k in charac_w_constant:
    h=h+1
    l=-1
    v=1
    result = scipy.stats.describe(Portfolios_lag.iloc[:,h], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics_w_lag.iloc[l+1,h]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics_w_lag.iloc[l+1,h]=q[1]
            else:
                if l==0:
                    Statistics_w_lag.iloc[l,h]=result[l]
                else:
                    Statistics_w_lag.iloc[l+1,h]=result[l]

Statistics_w_lag=Statistics_w_lag.append(Sharpe_ratio)
Statistics_w_lag=Statistics_w_lag.append(Beta)
#Export Statistics_w_lag
Statistics_w_lag.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_before_2010_lag.xlsx',index=True,header=True)
#Import Statistics_w_lag
# Statistics_w_lag=pd.read_excel(r'Statistics_before_2010_lag.xlsx')






################_PERIOD 2010 - 2019_##############

#Same but for the period 2010-2019
Portfolios_New=Portfolios.copy()
for t in Portfolios.index:
    if t.year==2008 or t.year==2009:
        Portfolios_New=Portfolios_New.iloc[1:]
        
Corr=Portfolios_New.corr()
Corr.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Correlations_After_2010.xlsx',index=True,header=True)

#Get Rf for 2010-2019
RF_New = RF.copy()
for t in RF_New.index:
    if t.year==2008 or t.year==2009:
        RF_New=RF_New.drop(t)
        
#Get Mkt for 2010-2019
Mkt_New = Mkt.copy()
for t in Mkt_New.index:
    if t.year==2008 or t.year==2009:
        Mkt_New=Mkt_New.drop(t)        
        
#Sharpe ratio

Sharpe_ratio = pd.DataFrame(columns=charac_w_constant, index=['Sharpe ratio'])
Excess_return = Portfolios_New.copy()
for i in charac_w_constant:
    Excess_return.loc[:,i] = Portfolios_New.loc[:,i] - RF_New.loc[:,'RF']/(100*12)
var = Portfolios_New.var()

for i in charac_w_constant:
    index = charac_w_constant.index(i)
    Sharpe_ratio.loc[:,i]= (12**0.5)*(Excess_return.loc[:,i].mean()/var[index]**0.5)
    
#Computation of beta
Beta =pd.DataFrame(columns=charac_w_constant, index=['Beta'])

for c in charac_w_constant:
    Cov_matrix=np.cov(Portfolios_New.loc[:,c],Mkt_New.loc[:,'Market'])
    Beta.loc['Beta',c]=(Cov_matrix[0][1])/Cov_matrix[1][1]

#Computation of Time Series Statistics
result = scipy.stats.describe(Portfolios_New.iloc[:,1], ddof=1, bias=False)
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=charac_w_constant)
h=-1
for k in charac_w_constant:
    h=h+1
    l=-1
    v=1
    result = scipy.stats.describe(Portfolios_New.iloc[:,h], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,h]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,h]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,h]=result[l]
                else:
                    Statistics.iloc[l+1,h]=result[l]

Statistics=Statistics.append(Sharpe_ratio)
Statistics=Statistics.append(Beta)
#Export Statistics
Statistics.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_after_2010.xlsx',index=True,header=True)
#Import Statistics
# Statistics = pd.read_excel(r'Statistics_after_2010.xlsx')

################################## With LAG: 2010 - 2019 ###################################

#Same but for the period 2010-2019
Portfolios_New_lag=Portfolios_lag.copy()
for t in Portfolios_lag.index:
    if t.year==2008 or t.year==2009:
        Portfolios_New_lag=Portfolios_New_lag.iloc[1:]
        
Corr_w_lag=Portfolios_New_lag.corr()
Corr_w_lag.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Correlations_After_2010_w_lag.xlsx',index=True,header=True)

#Sharpe ratio

Sharpe_ratio = pd.DataFrame(columns=charac_w_constant, index=['Sharpe ratio'])
Excess_return = Portfolios_New_lag.copy()
for i in charac_w_constant:
    Excess_return.loc[:,i] = Portfolios_New_lag.loc[:,i] - RF_New.loc[:,'RF']/(100*12)
var = Portfolios_New_lag.var()

for i in charac_w_constant:
    index = charac_w_constant.index(i)
    Sharpe_ratio.loc[:,i]= (12**0.5)*(Excess_return.loc[:,i].mean()/var[index]**0.5)
    
#Computation of beta
Beta =pd.DataFrame(columns=charac_w_constant, index=['Beta'])

for c in charac_w_constant:
    Cov_matrix=np.cov(Portfolios_New_lag.loc[:,c],Mkt_New.loc[:,'Market'])
    Beta.loc['Beta',c]=(Cov_matrix[0][1])/Cov_matrix[1][1]

#Computation of Time Series Statistics
result = scipy.stats.describe(Portfolios_New_lag.iloc[:,1], ddof=1, bias=False)
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=charac_w_constant)
h=-1
for k in charac_w_constant:
    h=h+1
    l=-1
    v=1
    result = scipy.stats.describe(Portfolios_New_lag.iloc[:,h], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,h]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,h]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,h]=result[l]
                else:
                    Statistics.iloc[l+1,h]=result[l]

Statistics_w_lag=Statistics.append(Sharpe_ratio)
Statistics_w_lag=Statistics_w_lag.append(Beta)
Statistics_w_lag.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_after_2010_w_lag.xlsx',index=True,header=True)



##########################################################################
##########################################################################

#Proofs

#Graph of Market vs Market Cap and BTMV
graph_prices(Portfolios_New,'Market Cap','b') 
Marketgg=pd.DataFrame(columns=['Market','Market Cap','BTMV'],index=Mkt_New.index)
Marketgg.loc[:,'Market']=Mkt_New.loc[:,'Market']
Marketgg.loc[:,['Market Cap','BTMV']]=Portfolios_New.loc[:,['Market Cap','BTMV']]      
graph_prices(Marketgg,['Market Cap','BTMV','Market'],'b')        

#Check for strange betas
plt.scatter(Mkt.loc[:,'Market'],Portfolios.loc[:,'Waste Total'])
plt.scatter(Mkt.loc[:,'Market'],Portfolios.loc[:,'Total Waste To Revenues USD in millions'])
plt.scatter(Mkt.loc[:,'Market'],Portfolios.loc[:,'Market Cap'])
plt.scatter(Mkt.loc[:,'Market'],Portfolios.loc[:,'BTMV'])

#Estimation of beta by linear regression model and creation of the graph
y=Portfolios_New.loc[:,'Market Cap']
x=Mkt_New.loc[:,'Market']
reg=LinearRegression().fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
z=reg.coef_[0][0]
y_p=reg.predict(np.array(x).reshape(-1,1))
plt.scatter(Mkt_New.loc[:,'Market'],Portfolios_New.loc[:,'Market Cap'])
plt.plot(Mkt_New.loc[:,'Market'],y_p)




##############################################################################
###########################        IPCA       #################################
##############################################################################

##########################  Period 2008 - 2019  ##############################

######################## Simpliest way to the PCA ##############################

Square_matrix = np.dot(Portfolios.iloc[:,0:len(Portfolios.columns)-1].transpose(),Portfolios.iloc[:,0:len(Portfolios.columns)-1])/len(Portfolios.index)
eigen_values_vectors = np.linalg.eig(Square_matrix)

Analysis = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios.columns[:-1])
for i in range(0,5):
    for j in range(0,len(Portfolios.columns)-1):
        Analysis.iloc[j,i] = eigen_values_vectors[1][j][i]
        
#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,4.8))
    plt.title('Period 2008 - 2019 \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis.index, Analysis.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(25,4.8))
    rang = Analysis.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()
    

######################## Dmean method for the PCA ##############################

df_dmean = Portfolios.copy()
for c in df_dmean.columns[:-1]: #Do not consider the last column 'Nb of Stock'
    m = df_dmean.loc[:,c].mean()
    df_dmean.loc[:,c] = df_dmean.loc[:,c] - m 
    
size=len(Portfolios.columns)-1

Square_matrix_2 = np.dot(df_dmean.iloc[:,0:size].transpose(),df_dmean.iloc[:,0:size])/len(df_dmean.index)
eigen_values_vectors_2 = np.linalg.eig(Square_matrix_2)
lamdba = eigen_values_vectors_2[1]*((len(Portfolios.columns)-1)**0.5)
h = np.dot(np.dot(np.linalg.inv(np.dot(lamdba.transpose(), lamdba)),lamdba.transpose()),Portfolios.iloc[:,0:size].transpose())
h = pd.DataFrame(data=h)

Analysis_2 = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios.columns[:-1])
for j in range(0,5):
    for i in range(0,len(Portfolios.columns[:-1])):
        Analysis_2.iloc[i,j] = lamdba[i][j]

#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,6))
    plt.title('Period 2008 - 2019 \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis_2.index, Analysis_2.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(15,4.8))
    rang = Analysis_2.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()


#Representation of the eigenvalues
eigenvalues = eigen_values_vectors_2[0]/eigen_values_vectors_2[0].mean()
plt.figure()
plt.title('Plot of the eigenvalues divided by the sum of the eigenvalues')
plt.axvline(x=5, ymin=0, ymax=30, color='r')
plt.plot(eigenvalues)

#Analysis on h sharpe ratio 
Sharpe_ratio_h = pd.DataFrame(index=['Sharpe ratio','Flip'], columns=(['1','2','3','4','5']))
for i in range(5):
    s=0
    if h.iloc[i,:].mean()<0:
        m = -h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'F'
    else:
        m = h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'N-F'


Sharpe_ratio_h.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Sharpe_ratio_h_before_2010.xlsx',index=True,header=True)




############################      WITH LAG       ###########################

df_dmean = Portfolios_lag.copy()
for c in df_dmean.columns[:-1]:
    m = df_dmean.loc[:,c].mean()
    df_dmean.loc[:,c] = df_dmean.loc[:,c] - m 

size = len(df_dmean.columns[:-1])

Square_matrix_2 = np.dot(df_dmean.iloc[:,0:size].transpose(),df_dmean.iloc[:,0:size])/len(df_dmean.index)
eigen_values_vectors_2 = np.linalg.eig(Square_matrix_2)
lamdba = eigen_values_vectors_2[1]*(size**0.5)
h = np.dot(np.dot(np.linalg.inv(np.dot(lamdba.transpose(),lamdba)),lamdba.transpose()),Portfolios_lag.iloc[:,0:size].transpose())
h = pd.DataFrame(data=h)

#Representation of the eigenvalues
eigenvalues = eigen_values_vectors_2[0]/eigen_values_vectors_2[0].mean()
plt.figure()
plt.title('Plot of the eigenvalues divided by the sum of the eigenvalues')
plt.axvline(x=5, ymin=0, ymax=30, color='r')
plt.plot(eigenvalues)

Analysis_2 = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios_lag.columns[:-1])
for j in range(0,5):
    for i in range(0,len(Portfolios_lag.columns[:-1])):
        Analysis_2.iloc[i,j] = eigen_values_vectors_2[1][i][j]

#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,6))
    plt.title('Period 2008 - 2019 with lag \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis_2.index, Analysis_2.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(15,4.8))
    rang = Analysis_2.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()

#Analysis on h sharpe ratio 
Sharpe_ratio_h = pd.DataFrame(index=['Sharpe ratio','Flip'], columns=(['1','2','3','4','5']))
for i in range(5):
    s=0
    if h.iloc[i,:].mean()<0:
        m = -h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF.iloc[j+1,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'F'
    else:
        m = h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF.iloc[j+1,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'N-F'

Sharpe_ratio_h.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Sharpe_ratio_h_before_2010_w_lag.xlsx',index=True,header=True)

Beta =pd.DataFrame(columns=(['1','2','3','4','5']), index=['Beta'])
Mkt_for_h = Mkt_lag.copy()
Mkt_for_h.reset_index(drop=True, inplace=True)
Mkt_for_h=Mkt_for_h.iloc[:-1,:]
for c in range(5):
    Cov_matrix=np.cov(h.iloc[c,:],Mkt_for_h.loc[:,'Market'])
    Beta.iloc[0,c]=(Cov_matrix[0][1])/Cov_matrix[1][1]

#Statistic for h
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=(['1','2','3','4','5']))
p=-1
for k in range(5):
    p=p+1
    l=-1
    v=1
    result = scipy.stats.describe(h.transpose().iloc[:,p], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,p]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,p]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,p]=result[l]
                else:
                    Statistics.iloc[l+1,p]=result[l]

Statistics=Statistics.append(Sharpe_ratio_h)
Statistics=Statistics.append(Beta)
Statistics.to_excel(r'C:\EDHEC\Cours\S2\S2_b\ERI\Python\Statistics_h_2008_lag.xlsx',index=True,header=True)




##############################################################################
##########################  Period 2010 - 2019  ##############################

######################## Simpliest way to the PCA ##############################

Square_matrix = np.dot(Portfolios_New.iloc[:,0:len(Portfolios_New.columns)-1].transpose(),Portfolios_New.iloc[:,0:len(Portfolios_New.columns)-1])/len(Portfolios_New.index)
eigen_values_vectors_2 = np.linalg.eig(Square_matrix)

Analysis = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios_New.columns[:-1])
for i in range(0,5):
    for j in range(0,len(Portfolios_New.columns)-1):
        Analysis.iloc[j,i] = eigen_values_vectors[1][j][i]
        
#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,4.8))
    plt.title('Period 2010 - 2019 \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis.index, Analysis.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(25,4.8))
    rang = Analysis.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()
    

######################## Dmean method for the PCA ##############################

df_dmean = Portfolios_New.copy()
for c in df_dmean.columns[:-1]:
    m = df_dmean.loc[:,c].mean()
    df_dmean.loc[:,c] = df_dmean.loc[:,c] - m
    
size = len(df_dmean.columns[:-1])
    

Square_matrix_2 = np.dot(df_dmean.iloc[:,0:size].transpose(),df_dmean.iloc[:,0:size])/len(df_dmean.index)
eigen_values_vectors_2 = np.linalg.eig(Square_matrix_2)
lamdba = eigen_values_vectors_2[1]*(size**0.5)
h = np.dot(np.dot(np.linalg.inv(np.dot(lamdba.transpose(),lamdba)),lamdba.transpose()),Portfolios_New.iloc[:,0:size].transpose())
h = pd.DataFrame(data=h)

#Representation of the eigenvalues
eigenvalues = eigen_values_vectors_2[0]/eigen_values_vectors_2[0].mean()
plt.figure()
plt.title('Plot of the eigenvalues divided by the sum of the eigenvalues')
plt.axvline(x=5, ymin=0, ymax=30, color='r')
plt.plot(eigenvalues)

Analysis_2 = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios_New.columns[:-1])
for j in range(0,5):
    for i in range(0,size):
        Analysis_2.iloc[i,j] = lamdba[i][j]

#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,6))
    plt.title('Period 2010 - 2019 \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis_2.index, Analysis_2.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(15,4.8))
    rang = Analysis_2.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()

#Analysis on h sharpe ratio 
Sharpe_ratio_h = pd.DataFrame(index=['Sharpe ratio','Flip'], columns=(['1','2','3','4','5']))
for i in range(5):
    s=0
    if h.iloc[i,:].mean()<0:
        m = -h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF_New.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'F'
    else:
        m = h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF_New.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'N-F'

Sharpe_ratio_h.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Sharpe_ratio_h_after_2010.xlsx',index=True,header=True)


############################      WITH LAG      ###########################

df_dmean = Portfolios_New_lag.copy()
for c in df_dmean.columns[:-1]:
    m = df_dmean.loc[:,c].mean()
    df_dmean.loc[:,c] = df_dmean.loc[:,c] - m

size=len(Portfolios_New_lag.columns)-1

Square_matrix_2 = np.dot(df_dmean.iloc[:,0:size].transpose(),df_dmean.iloc[:,0:size])/len(df_dmean.index)
eigen_values_vectors_2 = np.linalg.eig(Square_matrix_2)
lamdba = eigen_values_vectors_2[1]*(size**0.5)
h = np.dot(np.dot(np.linalg.inv(np.dot(lamdba,lamdba.transpose())),lamdba.transpose()),Portfolios_New_lag.iloc[:,0:size].transpose())
h = pd.DataFrame(data=h)

#Representation of the eigenvalues
eigenvalues = eigen_values_vectors_2[0]/eigen_values_vectors_2[0].mean()
plt.figure()
plt.title('Plot of the eigenvalues divided by the sum of the eigenvalues')
plt.axvline(x=5, ymin=0, ymax=30, color='r')
plt.plot(eigenvalues)

Analysis_2 = pd.DataFrame(columns=['1','2','3','4','5'], index=Portfolios_New_lag.columns[:-1])
for j in range(0,5):
    for i in range(0,size):
        Analysis_2.iloc[i,j] = eigen_values_vectors_2[1][i][j]

#Plot the grpah for the first five eigenvalues
for i in range(5):
    plt.figure(figsize=(25,6))
    plt.title('Period 2010 - 2019 with lag \n Barplot for the eigenvector number '+str(i+1), fontdict = {'fontsize' : 25})
    plt.bar(Analysis_2.index, Analysis_2.iloc[:,i])
    plt.xticks(rotation=90)
    plt.show()

#Rank to plot only the first four
for i in range(5):
    plt.figure(figsize=(15,4.8))
    rang = Analysis_2.iloc[:,i].abs().sort_values(ascending=False)
    plt.bar(rang.index[0:3], rang.iloc[0:3])
    plt.show()
    


#Analysis on h sharpe ratio 
Sharpe_ratio_h = pd.DataFrame(index=['Sharpe ratio','Flip'], columns=(['1','2','3','4','5']))
for i in range(5):
    s=0
    if h.iloc[i,:].mean()<0:
        m = -h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF_New.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'F'
    else:
        m = h.iloc[i,:]
        for j in range(len(m)):
            s=m[j]-RF_New.iloc[j,0]/(100*12)
        Sharpe_ratio_h.iloc[0,i] = (s.mean()/((m.var())**0.5))*(12**0.5)
        Sharpe_ratio_h.iloc[1,i] = 'N-F'

Beta = pd.DataFrame(columns=(['1','2','3','4','5']), index=['Beta'])

for c in range(5):
    Cov_matrix=np.cov(h.iloc[c,:],Mkt_New.loc[:,'Market'])
    Beta.iloc[0,c]=(Cov_matrix[0][1])/Cov_matrix[1][1]


#Statistic for h
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=(['1','2','3','4','5']))
p=-1
for k in range(5):
    p=p+1
    l=-1
    v=1
    result = scipy.stats.describe(h.transpose().iloc[:,p], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,p]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,p]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,p]=result[l]
                else:
                    Statistics.iloc[l+1,p]=result[l]

Statistics=Statistics.append(Sharpe_ratio_h)
Statistics=Statistics.append(Beta)
Statistics.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_h_2010_lag.xlsx',index=True,header=True)





##############################################################################
########################## ESG ETF Analys ####################################
##############################################################################
ESG_ETF=returns(ESG)
ESG_ETF.drop(ESG_ETF.head(1).index,inplace=True)

#Computation of Sharpe ratio
Sharpe_ratio = pd.DataFrame(columns=ESG_ETF.columns, index=['Sharpe ratio'])
Excess_return = ESG_ETF.copy()
for i in ESG_ETF.columns:
    for t in ESG_ETF.index:
        Excess_return.loc[t,i] = Excess_return.loc[t,i] - RF.loc[t,'RF']/(100*12) 
var = ESG_ETF.var()

for i in ESG_ETF.columns:
    Sharpe_ratio.loc[:,i]= (12**0.5)*(Excess_return.loc[:,i].mean()/var[i]**0.5)
    
    
Beta =pd.DataFrame(columns=ESG_ETF.columns, index=['Beta'])

Mkt_ESG=ESG_ETF.iloc[:,0].copy()
for t in ESG_ETF.index:
    Mkt_ESG[t]=Mkt.loc[t,'Market']


for c in ESG_ETF.columns:
    Cov_matrix=np.cov(ESG_ETF.loc[:,c],Mkt_ESG.loc[:])
    Beta.loc['Beta',c]=(Cov_matrix[0][1])/Cov_matrix[1][1]  
    
#Computation of Time Series Statistics
Stats=['Observations','Min','Max','Mean','Variance','Skewness','Kurtosis']
Statistics=pd.DataFrame(index=Stats,columns=ESG_ETF.columns)
h=-1
for k in ESG_ETF.columns:
    h=h+1
    l=-1
    v=1
    result = scipy.stats.describe(ESG_ETF.iloc[:,h], ddof=1, bias=False)
    for j in Stats:
        l=l+1
        if l==1 and v==1:
            l=l-1
            q=result[l+1]
            Statistics.iloc[l+1,h]=q[0]
            v=v+1
        else:
            if l==1:
                q=result[l]
                Statistics.iloc[l+1,h]=q[1]
            else:
                if l==0:
                    Statistics.iloc[l,h]=result[l]
                else:
                    Statistics.iloc[l+1,h]=result[l]
                    

Statistics=Statistics.append(Sharpe_ratio)
Statistics=Statistics.append(Beta)
Statistics.to_excel(r'C:\Users\apala\OneDrive\Documents\EDHEC\Second Term\ERI\ERI-Climate risk\Python\Statistics_ESG_ETF.xlsx',index=True,header=True)
