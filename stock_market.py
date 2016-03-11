
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from pandas import Series,DataFrame


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')


# In[3]:

from pandas.io.data import DataReader
from datetime import datetime


# In[4]:

tech_list = ['TXN','NXPI','STM','ADI']


# In[5]:

end=datetime.now()
start=datetime(end.year-2,end.month,end.day)


# In[6]:

for stock in tech_list:
    globals()[stock] = DataReader(stock,'yahoo',start,end)


# In[7]:

ADI.describe()


# In[8]:

ADI.info()


# In[9]:

ADI['Adj Close'].plot(legend=True,figsize=(10,4))


# In[10]:

ADI['Volume'].plot(legend=True,figsize=(10,4))


# In[11]:

ma_day = [10,20,50]
for ma in ma_day:
    column_name = 'MA for %s days'%(str(ma))
    ADI[column_name]=pd.rolling_mean(ADI['Adj Close'],ma)


# In[12]:

ADI[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))


# In[13]:

ADI['Daily Return']=ADI['Adj Close'].pct_change()


# In[14]:

ADI['Daily Return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[15]:

sns.distplot(ADI['Daily Return'].dropna(),bins=100,color='blue')


# In[16]:

ADI['Daily Return'].hist(bins=100)


# In[17]:

closing_df = DataReader(tech_list,'yahoo',start,end)['Adj Close']


# In[18]:

closing_df.head()


# In[19]:

tech_rets = closing_df.pct_change()


# In[20]:

tech_rets.head()


# In[21]:

sns.jointplot('ADI','ADI',tech_rets,kind='scatter',color='seagreen')


# In[22]:

sns.jointplot('TXN','ADI',tech_rets,kind='scatter',color='darkblue')


# In[23]:

tech_rets.head()


# In[24]:

sns.pairplot(tech_rets.dropna())


# In[25]:

returns_fig = sns.PairGrid(tech_rets.dropna())
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[26]:

returns_fig = sns.PairGrid(closing_df)
returns_fig.map_upper(plt.scatter,color='purple')
returns_fig.map_lower(sns.kdeplot,cmap='cool_d')
returns_fig.map_diag(plt.hist,bins=30)


# In[27]:

sns.corrplot(tech_rets.dropna(),annot=True)


# In[28]:

sns.corrplot(closing_df,annot=True)


# In[29]:

rets = tech_rets.dropna()
area = np.pi*30
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')
for label,x,y in zip(rets.columns,rets.mean(),rets.std()):
    plt.annotate(
    label,
    xy = (x,y), xytext = (50,50),
    textcoords = 'offset points',ha='right', va = 'bottom',
    arrowprops = dict(arrowstyle = '-',connectionstyle='arc3,rad=-0.3'))


# In[30]:

sns.distplot(ADI['Daily Return'].dropna(),bins=100,color='purple')


# In[31]:

rets.head()


# Value At Risk

# In[32]:

rets['ADI'].quantile(0.05)


# Monte Carlo

# In[33]:

days = 5*365
dt = 1/365
mu = rets.mean()['ADI']
sigma = rets.std()['ADI']


# In[34]:

def stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x] = mu*dt
        price[x] = price[x-1] + (price[x-1]*drift[x]+shock[x])
    return price


# In[35]:

ADI.head()


# In[36]:

start_price = 51.85
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for ADI')


# In[37]:

runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


# In[38]:

q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
plt.figtext(0.68,0.8,s='Start Price: $%.2f'%start_price)
plt.figtext(0.68,0.7,"Mean final price: $%2.f"%simulations.mean())
plt.figtext(0.68,0.6,"VaR(0.99): $%.2f"%(start_price-q,))
plt.figtext(0.15,0.6,"q(0.99): $%.2f"%q)
plt.axvline(x=q,linewidth=4,color='r')
plt.title(u'Final price distribution for ADI after %s days'%days,weight='bold')


# In[ ]:



