# In[4]:


#from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from scipy.stats import pointbiserialr

import tensorflow as tf


# In[6]:


hurricane_data = './Datasets/Atlantic Hurricane ACE Data.csv'
hurricane_data = pd.read_csv(hurricane_data)
ace = hurricane_data.ACE
number_of_storms = hurricane_data.Named_Storms

hurricaneYear = hurricane_data.Year
hurricaneTime = hurricane_data.time

features = ['Year', 'time']
hurricaneX = hurricane_data[features]


# # ML Pipeline

# In[ ]:


#import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error
import math
#import statsmodels.api as sm
#from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
plt.style.use('dark_background')

params = {'n_estimators': 10000, 'max_depth': 4,
          'learning_rate': 0.1, 'loss': 'ls'}
model = GradientBoostingRegressor(**params)
modelSVR = SVR(kernel='rbf', C=1e4, epsilon=0.1)
#X_train = x_train = X_test = x_test = y_train = y_test = None

class MachineLearningAlgorithm:
    def __init__(self, x, X, y, n):
        print('Additive Gradient Boosting Learning Model: ')
        self.x = x
        self.X = X
        self.y = y
        self.option = n
        
    def testSplitValid(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        
        '''sc_X = StandardScaler()
        sc_y = StandardScaler()
        X = sc_X.fit_transform(x)
        y = sc_y.fit_transform(y)'''
        
        plt.scatter(x_train, y_train, color='r', s = 6, alpha=0.5)
        plt.scatter(x_test, y_test, color='b', s = 6, alpha=0.5)
        plt.title('Train/Test Split Graph')
        plt.xlabel('Years')
        plt.ylabel('Molar Ratio')
        plt.show()
        print(' ')
        if self.option == 1:
            self.gradientBoosting(X_train, x_train, y_train, X_test, x_test, y_test)
        if self.option == 2:
            self.supportVector(X_train, x_train, y_train, X_test, x_test, y_test)
        if self.option == 3:
            self.arima(X_train, x_train, y_train, X_test, x_test, y_test)
            
    def arima(X_train, x_train, y_train, X_test, x_test, y_test):
        result = seasonal_decompose(self.y, model ='multiplicative')
        print(result.plot())
        
                
        
        
    
    def supportVector(self, X_train, x_train, y_train, X_test, x_test, y_test):
        print('SVR Model Training: ')
        modelSVR.fit(X_train, y_train)
        vectors = modelSVR.support_vectors_
        indices = [modelSVR.support_]
        
        vector_value = []
        vector_year = []
        for i in vectors:
            vector_value.append(i[0])
            vector_year.append(i[1])
            
        train_theoretical = modelSVR.predict(X_train)
        plt.scatter(x_train, y_train, color="g", s = 6, alpha=0.5)
        plt.scatter(x_train, train_theoretical, color='y', s = 6, alpha=0.5)
        plt.scatter(vector_value, vector_year, color='r', s = 2, alpha=0.5)
        plt.show()
        print('MAE for train set: ', mean_absolute_error(y_train, train_theoretical))
        print('RMSE for train set: ', math.sqrt(mean_squared_error(y_train, train_theoretical)))
        print(' ')
        
        print("Validating Test Data: ")
        test_theoretical = modelSVR.predict(X_test)
        plt.scatter(x_test, y_test, color="g", s = 6, alpha=0.5)
        plt.scatter(x_test, test_theoretical, color='y', s = 6, alpha=0.5)
        plt.show()
        print('MAE for test set: ', mean_absolute_error(y_test, test_theoretical))
        print('RMSE for test set: ', math.sqrt(mean_squared_error(y_test, test_theoretical)))
        print(' ')
        print(' ')
        
        '''mae_list = []
        max_leaf_node_list = [5, 50, 500, 5000]
        for max_leaf_nodes in max_leaf_node_list:
            my_mae = self.get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
            mae_list.append(my_mae)
            print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
            print(' ')
        
        mae_list.sort()
        smallest_mae = mae_list[0]
        element_index = mae_list.index(smallest_mae)
        final_mln = max_leaf_node_list[element_index]
        self.finalFit(final_mln)'''
    def gradientBoosting(self, X_train, x_train, y_train, X_test, x_test, y_test):
        print('GB Model Training: ')
        model.fit(X_train, y_train)
        train_theoretical = model.predict(X_train)
        plt.scatter(x_train, y_train, color="g", s = 6, alpha=0.5)
        plt.scatter(x_train, train_theoretical, color='y', s = 6, alpha=0.5)
        plt.show()
        print('MAE for train set: ', mean_absolute_error(y_train, train_theoretical))
        print(' ')
        
        print("Validating Test Data: ")
        test_theoretical = model.predict(X_test)
        plt.scatter(x_test, y_test, color="g", s = 6, alpha=0.5)
        plt.scatter(x_test, test_theoretical, color='y', s = 6, alpha=0.5)
        plt.show()
        print('MAE for test set: ', mean_absolute_error(y_test, test_theoretical))
        print(' ')
        print(' ')
        
        mae_list = []
        max_leaf_node_list = [5, 50, 500, 5000]
        for max_leaf_nodes in max_leaf_node_list:
            my_mae = self.get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
            mae_list.append(my_mae)
            print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
            print(' ')
        
        mae_list.sort()
        smallest_mae = mae_list[0]
        element_index = mae_list.index(smallest_mae)
        final_mln = max_leaf_node_list[element_index]
        self.finalFit(final_mln)
        return
        
    def get_mae(self, mln, train_X, test_X, train_y, test_y):
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        mae = mean_absolute_error(test_y, preds)
        return(mae)
    
    def finalFit(self, mln):
        self.final_model = GradientBoostingRegressor(max_leaf_nodes=mln, random_state=0)
        self.final_model.fit(self.X, self.y)
        theoretical_preds = self.final_model.predict(self.X)
        print('Final Holistic Prediction: ')
        plt.scatter(self.x, self.y, color="g", s = 6, alpha=0.5)
        plt.scatter(self.x, theoretical_preds, color='y', s = 6, alpha=0.5)
        plt.show()
        print('MAE for final prediction: ', mean_absolute_error(self.y, theoretical_preds))
        return
        
    def predictGB(self, x, X):
        final_preds = self.final_model.predict(X)
        plt.scatter(self.x, self.y, color="g", s = 6, alpha=0.5)
        plt.scatter(x, final_preds, color='y', s = 6, alpha=0.5)
        return final_preds
        


# # ACE Data

# In[ ]:


plt.plot(hurricaneYear, ace, color='y')
plt.scatter(hurricaneYear, ace, color='r', s = 8, alpha=0.8)
#plt.scatter(hurricaneYear, number_of_storms, color='g', s = 8, alpha=0.8)
plt.title('Accumulated Cyclone Energy Index Measurements (ACE)')
plt.xlabel('Years since 1851')
plt.ylabel('ACE (10^4 kt^2)')


# ### Data Smoothing and Downscaling

# In[ ]:


from scipy.interpolate import splev, splrep

x_smooth = np.linspace(hurricaneYear.min(), hurricaneYear.max(), 1000)
x_smoothTime = np.linspace(hurricaneTime.min(), hurricaneTime.max(), 1000)
data = {'Year': x_smooth,
        'time': x_smoothTime
        }
smoothX = pd.DataFrame(data, columns = ['Year', 'time'])

spl = splrep(hurricaneYear, ace)
y_smooth = splev(x_smooth, spl)

plt.plot(hurricaneYear, ace, color='y')
plt.scatter(hurricaneYear, ace, color='r', s = 8, alpha=0.8)
plt.plot(x_smooth, y_smooth, color='g')
#plt.scatter(hurricaneYear, number_of_storms, color='g', s = 8, alpha=0.8)
plt.title('Accumulated Cyclone Energy Index Measurements (ACE)')
plt.xlabel('Years since 1851')
plt.ylabel('ACE (10^4 kt^2)')
plt.show()

#--------
#y_smooth = pd.DataFrame({'Y': y_smooth})

def movAvg(values, window):
    weights = np.repeat(1.0, window) /window
    smas = np.convolve(values,weights,'valid')
    return smas

def expMovAvg(values, window):
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum()
    
    a = np.convolve(values,weights)[:len(values)]
    a[:window]=a[window]
    return a

y_sma = movAvg(y_smooth, 25)
y_ema = expMovAvg(y_smooth, 25)
data = np.array([y_sma, y_ema[:976]])
y_mean = np.average(data, axis=0)

moving_average = np.concatenate((y_sma[:24], y_ema[24:]))

plt.plot(hurricaneYear, ace, color='y')
#plt.scatter(hurricaneYear, ace, color='r', s = 8, alpha=0.8)
plt.plot(x_smooth, y_smooth, color='g')
plt.plot(x_smooth[:976], y_sma, color='c')
plt.plot(x_smooth, y_ema, color='m')
plt.plot(x_smooth[:976], y_mean, color='r')
#plt.scatter(hurricaneYear, number_of_storms, color='g', s = 8, alpha=0.8)
plt.title('Accumulated Cyclone Energy Index Measurements (ACE)')
plt.xlabel('Years since 1851')
plt.ylabel('ACE (10^4 kt^2)')
plt.show()

plt.plot(hurricaneYear, ace, color='y')
#plt.scatter(hurricaneYear, ace, color='r', s = 8, alpha=0.8)
plt.plot(x_smooth, y_smooth, color='g')
plt.plot(x_smooth, moving_average, color='y')
#plt.scatter(hurricaneYear, number_of_storms, color='g', s = 8, alpha=0.8)
plt.title('Accumulated Cyclone Energy Index Measurements (ACE)')
plt.xlabel('Years since 1851')
plt.ylabel('ACE (10^4 kt^2)')
plt.show()


# In[ ]:


hurricaneYearExtrap = list(range(1820, 2020))
hurricaneTimeExtrap = [ x-1820 for x in hurricaneYearExtrap ]

data = {'Year': hurricaneYearExtrap,
        'time': hurricaneTimeExtrap
        }
hurricaneExtrapX = pd.DataFrame(data, columns = ['Year', 'time'])


# In[ ]:


ace_boostingModel = MachineLearningAlgorithm(x_smooth, smoothX, y_smooth, 2)
ace_boostingModel.testSplitValid()
#ace_preds = ace_boostingModel.predictGB(hurricaneYearExtrap, hurricaneExtrapX)


# In[ ]:


north_tropicsModel = MachineLearningAlgorithm(SSTyear, SSTx, north_sst)
north_tropicsModel.testSplitValid()
north_tropics_preds = north_tropicsModel.predict(sstYearExtrap, sstExtrapX)


# In[ ]:


south_tropicsModel = MachineLearningAlgorithm(SSTyear, SSTx, south_sst)
south_tropicsModel.testSplitValid()
south_tropics_preds = south_tropicsModel.predict(sstYearExtrap, sstExtrapX)


# In[ ]:


optimal_tropics_anomaly = []
optimal_tropics_SST = []
for x in range(len(mid_tropics_preds)):
    avg = (mid_tropics_preds[x] + north_tropics_preds[x] + south_tropics_preds[x])/3
    optimal_tropics_anomaly.append(avg)
    optimal_tropics_SST.append(avg+27.69221833)


# In[ ]:


extrapolatedSST = pd.DataFrame({"Year":sstYearExtrap, 
                    "time":sstTimeExtrap,  
                    "extrapAnomaly":optimal_tropics_anomaly,
                    "extrapSST":optimal_tropics_SST}) 

extrapolatedSST.to_csv('./Datasets/Programmed Datasets/FinalSST.csv')


# ## End

# In[ ]:


CO2merge = []  #This is used to merge all of the CO2 dataset y-values into one dataset
CO2timeset = []  #This dataset will merge all of the years or x-values together

class order:
    def rank (self, a, b, c):
        originalRank = [CO2time[a], GHGt[b], NOAA_GHGtime[c]]
        rank = [CO2time[a], GHGt[b], NOAA_GHGtime[c]]
        rank.sort()
        least = rank[0]
        elementId = originalRank.index(least)
        
        if elementId == 0:
            CO2merge.append(CO2[a])
            a = a+1
        if elementId == 1:
            CO2merge.append(CO2_1[b])
            b = b+1
        if elementId == 2:
            CO2merge.append(CO2_2[c])
            c = c+1
            
        return a, b, c
        

order = order()

a = b = c = 0
while a <= len(CO2year) or b <= len(GHGyear) or c <= len(NOAA_GHGyear):
    '''order.rank(a, b, c)
    a = a
    b = b
    c = c'''
    if b == len(GHGyear):
        GHGt[b] = 1000
    if c == len(NOAA_GHGyear):
        NOAA_GHGtime[c] = 1000
        
    if a == len(CO2year):
        break
    
    originalRank = [CO2time[a], GHGt[b], NOAA_GHGtime[c]]
    rank = [CO2time[a], GHGt[b], NOAA_GHGtime[c]]
    rank.sort()
    least = rank[0]
    elementId = originalRank.index(least)
     
    if elementId == 0:
        CO2merge.append(CO2[a])
        CO2timeset.append(CO2time[a])
        a = a+1
    if elementId == 1:
        CO2merge.append(CO2_1[b])
        CO2timeset.append(GHGt[b])
        b = b+1
        
    if elementId == 2:
        CO2merge.append(CO2_2[c])
        CO2timeset.append(NOAA_GHGtime[c])
        c = c+1

years = []
for i in CO2timeset:
    years.append(i+1750)

data = {'Year': years,
        'time': CO2timeset,
        'CO2': CO2merge
        }
completeCO2set = pd.DataFrame(data, columns = ['Year', 'time', 'CO2'])
completeCO2set.to_csv('./Datasets/Programmed Datasets/HolisticCO2.csv')


# ## End

# In[ ]:


CO2new = pd.read_csv('./Datasets/Programmed Datasets/HolisticCO2.csv')
seasonalData = CO2new.CO2
seasonalTime = CO2new.time

seasonalData = seasonalData[42:]
seasonalTime = seasonalTime[42:]
seasonalT = []
for i in seasonalTime:
    seasonalT.append(i+1750)


rawData = pd.read_csv('./Datasets/Programmed Datasets/HolisticCO2.csv', header=0, index_col=0)
CO2seasonal = rawData[42:]

plt.style.use('ggplot')

decomposition = seasonal_decompose(CO2seasonal, model='additive', freq=1)

trend = decomposition.trend
seasonal = decomposition.seasonal 
residual = decomposition.resid

trend.plot()
seasonal.plot()
residual.plot()
plt.show()


# In[ ]:


def sma(value, window):
    weight = np.repeat(1.0, window)/window
    sma = np.convolve(value, weight, 'valid')
    return sma

smaCO2 = sma(seasonalData, 30)
smaTime = sma(seasonalT, 30)

from statsmodels.tsa.stattools import adfuller
rolmean = CO2seasonal.rolling(12).mean()
rolstd = CO2seasonal.rolling(12).std()

plt.plot(CO2seasonal['CO2'], color='green', label = 'Rolling Std')
plt.plot(rolmean['CO2'], color='red', label='Rolling Mean')
plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.plot()
plt.title('Carbon Dioxide Concentrations')
plt.xlabel('Years since 1950')
plt.ylabel('Parts per Million')

#print(rolmean)


# In[ ]:





# In[ ]:




