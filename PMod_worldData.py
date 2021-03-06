# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:12:33 2020

@author: jacka
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from scipy.optimize import curve_fit
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
from scipy.optimize import curve_fit
import math
import datetime


df = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')
col_name = ['Date', 'Country/Region', 'Province/State', 'Lat', 'Long', 'Confirmed', 'Recovered', 'Deaths', 'date']
newAgain = pd.DataFrame(columns = col_name)

rose = []
test = list(df['Country/Region'].unique())

for i in test:
    niles = df.groupby('Country/Region').get_group(i)
    rose.append(niles)
for r in rose:
    hold=[]
    timeD = r['Date'].tolist()
    for t in timeD:
        delt = datetime.datetime.strptime(t, "%Y-%m-%d").strftime("%m%d%Y")
        hold.append(delt)
    r['date'] = hold
    newAgain = pd.concat([newAgain, r])

newAgain = newAgain.drop(columns = 'Date').astype({'date':'float64'})
newAgain = newAgain.reindex(columns = ['date', 'Country/Region', 'Province/State', 'Lat', 'Long', 'Confirmed', 'Recovered', 'Deaths'])
print(newAgain)

filtered_data = newAgain[newAgain["Country/Region"] =='US']
US = filtered_data.drop(columns = ['Province/State','Lat', 'Long']) 
world_data = newAgain[newAgain["Country/Region"] =='France']
world_data1 = newAgain[newAgain["Country/Region"] =='China']
world_data2 = newAgain[newAgain["Country/Region"] =='Italy']
world_data3 = newAgain[newAgain["Country/Region"] =='France']
world_data4 = newAgain[newAgain["Country/Region"] =='England']
world_data5 = newAgain[newAgain["Country/Region"] =='US']
#world = world_data.drop(columns = ['Province/State','Lat', 'Long'])
#world = pd.concat([world_data, world_data1, world_data2, world_data3, world_data4, world_data5])
world = world.drop(columns = ['Province/State','Lat', 'Long'])
world= world.fillna(0)

#world.concat([world_data, world_data1])
print(US)
#print(world)

x = filtered_data.Confirmed.values.reshape(-1,1)
y = filtered_data.Deaths.values.reshape(-1,1)

xW = world.Confirmed.values.reshape(-1,1)
yW = world.Deaths.values.reshape(-1,1)

# To Predict Number of Deaths from Confirmed Cases:
train_xW, test_xW, train_yW, test_yW = train_test_split (xW, yW, test_size = 0.25, random_state = 1)
linear_model = LinearRegression()
linear_model.fit(train_xW, train_yW)

intercept = linear_model.intercept_
coeff = linear_model.coef_

test_prediction = linear_model.predict(test_xW)

# df_model = pd.DataFrame({'Actual':test_y1.flatten(), 
                        #'Predicted':test_prediction.flatten()})
# df_model = df_model.sort_values(by = ['coeff'])
# df_model

plt.title("Prediction Model") 
plt.xlabel("Confirmed Cases") 
plt.ylabel("Deaths") 
plt.plot(x, y, label = 'US Actual')
#plt.plot(xW, yW, label = 'Other country Actual')
plt.plot(test_xW, test_prediction, label = 'Predicted') 
plt.legend(loc = 'upper left')
plt.show()
# df_model.plot(x = 'Confirmed', y = 'coeff', kind = 'bar', figsize = (15, 10))
# fdf = pd.concat([test_x, test_y1], 1)
# fdf['Predicted'] = np.round(test_prediction, 1)
# fdf['Prediction_Error_Confirmed'] = fdf['Deaths'] - fdf['Predicted'] 
# fdf


world_data2 = newAgain[newAgain["Country/Region"] =='Italy']
world2 = world_data2.drop(columns = ['Province/State','Lat', 'Long'])
world2= world2.fillna(0)
xW2 = world2.Confirmed.values.reshape(-1,1)
yW2 = world2.Deaths.values.reshape(-1,1)
print(world2)

train_xW2, test_xW2, train_yW2, test_yW2 = train_test_split (xW2, yW2, test_size = 0.25, random_state = 1)
linear_model = LinearRegression()
linear_model.fit(train_xW2, train_yW2)

intercept = linear_model.intercept_
coeff = linear_model.coef_
test_prediction = linear_model.predict(test_xW2)

plt.title("Prediction Model") 
plt.xlabel("Confirmed Cases") 
plt.ylabel("Deaths") 
plt.plot(xW2, yW2, label = 'Italy Actual')
#plt.plot(xW, yW, label = 'Other country Actual')
plt.plot(test_xW2, test_prediction, label = 'Predicted') 
plt.legend(loc = 'upper left')
plt.show()


world_data3 = newAgain[newAgain["Country/Region"] =='Spain']
world3 = world_data3.drop(columns = ['Province/State','Lat', 'Long'])
world3= world3.fillna(0)
xW3 = world3.Confirmed.values.reshape(-1,1)
yW3 = world3.Deaths.values.reshape(-1,1)
print(world3)

train_xW3, test_xW3, train_yW3, test_yW3 = train_test_split (xW3, yW3, test_size = 0.25, random_state = 1)
linear_model = LinearRegression()
linear_model.fit(train_xW3, train_yW3)

intercept = linear_model.intercept_
coeff = linear_model.coef_
test_prediction = linear_model.predict(test_xW3)

plt.title("Prediction Model") 
plt.xlabel("Confirmed Cases") 
plt.ylabel("Deaths") 
plt.plot(xW3, yW3, label = 'Spain Actual')
#plt.plot(xW, yW, label = 'Other country Actual')
plt.plot(test_xW3, test_prediction, label = 'Predicted') 
plt.legend(loc = 'upper left')
plt.show()



world_data4 = newAgain[newAgain["Country/Region"] =='Japan']
world4 = world_data4.drop(columns = ['Province/State','Lat', 'Long'])
world4= world4.fillna(0)
xW4 = world4.Confirmed.values.reshape(-1,1)
yW4 = world4.Deaths.values.reshape(-1,1)
print(world4)

train_xW4, test_xW4, train_yW4, test_yW4 = train_test_split (xW4, yW4, test_size = 0.25, random_state = 1)
linear_model = LinearRegression()
linear_model.fit(train_xW4, train_yW4)

intercept = linear_model.intercept_
coeff = linear_model.coef_
test_prediction = linear_model.predict(test_xW4)

plt.title("Prediction Model") 
plt.xlabel("Confirmed Cases") 
plt.ylabel("Deaths") 
plt.plot(xW4, yW4, label = 'Japan Actual')
#plt.plot(xW, yW, label = 'Other country Actual')
plt.plot(test_xW4, test_prediction, label = 'Predicted') 
plt.legend(loc = 'upper left')
plt.show()




