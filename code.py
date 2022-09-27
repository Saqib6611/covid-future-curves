import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline
# filter warnings
import warnings
warnings.filterwarnings('ignore')
# upload datasets
confirmed_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_reported = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/07-18-2022.csv')
# analyze and clean data
latest_data.isna().sum(axis=0)
latest_data['Recovered']
# Fetching all the columns from confirmed dataset
cols = confirmed_cases.keys()
cols
# Extracting the date columns
confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]
deaths = deaths_reported.loc[:, cols[4]:cols[-1]]
recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]
recoveries1 = recovered_cases.loc[:, cols[4]:cols[-355]]

dates1= recoveries1.keys() 
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 

china_cases = [] 
italy_cases = []
us_cases = [] 
spain_cases = [] 
france_cases = [] 
germany_cases = [] 
uk_cases = [] 
russia_cases = []
india_cases = []

china_deaths = [] 
italy_deaths = []
us_deaths = [] 
spain_deaths = [] 
france_deaths = [] 
germany_deaths = [] 
uk_deaths = [] 
russia_deaths = []
india_deaths = []


china_recoveries = [] 
italy_recoveries = []
us_recoveries = [] 
spain_recoveries = [] 
france_recoveries = [] 
germany_recoveries = [] 
uk_recoveries = [] 
russia_recoveries = [] 
india_recoveries = []

china_recoveries1 = [] 
italy_recoveries1 = []
us_recoveries1 = [] 
spain_recoveries1 = [] 
france_recoveries1 = [] 
germany_recoveries1 = [] 
uk_recoveries1 = [] 
russia_recoveries1 = [] 
india_recoveries1 = []

for j in dates1:
  recovered_sum1 = recoveries[j].sum()
  china_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='China'][j].sum())
  italy_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Italy'][j].sum())
  us_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='US'][j].sum())
  spain_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Spain'][j].sum())
  france_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='France'][j].sum())
  germany_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Germany'][j].sum())
  uk_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='United Kingdom'][j].sum())
  russia_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='Russia'][j].sum())
  india_recoveries.append(recovered_cases[recovered_cases['Country/Region']=='India'][j].sum())
  #print(recovered_sum)
print(recovered_sum1)


for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    #recovered_sum = recoveries[i].sum()
    
    
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum1)
    total_active.append(confirmed_sum-death_sum-recovered_sum1)
    
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum1/confirmed_sum)

    china_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='France'][i].sum())
    germany_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Germany'][i].sum())
    uk_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='Russia'][i].sum())
    india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())
    
    
    china_deaths.append(deaths_reported[deaths_reported['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_reported[deaths_reported['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_reported[deaths_reported['Country/Region']=='France'][i].sum())
    germany_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Germany'][i].sum())
    uk_deaths.append(deaths_reported[deaths_reported['Country/Region']=='United Kingdom'][i].sum())
    russia_deaths.append(deaths_reported[deaths_reported['Country/Region']=='Russia'][i].sum())
    india_deaths.append(deaths_reported[deaths_reported['Country/Region']=='India'][i].sum())
    
    
    china_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='China'][i].sum())
    italy_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='Italy'][i].sum())
    us_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='US'][i].sum())
    spain_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='Spain'][i].sum())
    france_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='France'][i].sum())
    germany_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='Germany'][i].sum())
    uk_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='United Kingdom'][i].sum())
    russia_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='Russia'][i].sum())
    india_recoveries1.append(recovered_cases[recovered_cases['Country/Region']=='India'][i].sum())

# Total number of cases, deaths and recoveries.
total_Data_df = pd.DataFrame({'Total confirmed': [confirmed_sum], 'total Deaths': [death_sum], 'Total Recoveries': [recovered_sum1]})
total_Data_df.style.background_gradient(cmap='Blues_r')

# total cases till date of top affected countries/region
Country_cases_df = pd.DataFrame({'China': china_cases, 'Italy': italy_cases, 'USA': us_cases,
                          'Spain': spain_cases, 'France' : france_cases, 'Germany': germany_cases, 'UK': uk_cases,
                          'India': india_cases})
Country_cases_df.style.background_gradient(cmap='Blues')                

# total deaths till date of top affected countries/region
Country_deaths_df = pd.DataFrame({'China': china_deaths, 'Italy': italy_deaths, 'USA': us_deaths,
                          'Spain': spain_deaths, 'France' : france_deaths, 'Germany': germany_deaths, 'UK': uk_deaths,
                          'India': india_deaths})
Country_deaths_df.style.background_gradient(cmap='Blues_r')

# total Recoveries till date of top affected countries/region
Country_recoveries_df = pd.DataFrame({'China': china_recoveries, 'Italy': italy_recoveries, 'USA': us_recoveries,
                          'Spain': spain_recoveries, 'France' : france_recoveries, 'Germany': germany_recoveries, 'UK': uk_recoveries,
                          'India': india_recoveries})
Country_recoveries_df.style.background_gradient(cmap='Blues_r')

def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d

def daily_increase1(data1):
    d = [] 
    for i in range(len(data1)):
        if i == 0:
            d.append(data1[0])
        else:
            d.append(data1[i]-data1[i-1])
    return d  

# confirmed cases
world_daily_increase = daily_increase(world_cases)
china_daily_increase = daily_increase(china_cases)
italy_daily_increase = daily_increase(italy_cases)
us_daily_increase = daily_increase(us_cases)
spain_daily_increase = daily_increase(spain_cases)
france_daily_increase = daily_increase(france_cases)
germany_daily_increase = daily_increase(germany_cases)
uk_daily_increase = daily_increase(uk_cases)
india_daily_increase = daily_increase(india_cases)

# number of increased cases per day in top affected countries/region
Increase_pattern_df = pd.DataFrame({'World increase': world_daily_increase, 'china': china_daily_increase, 'Italy': italy_daily_increase,
                          'USA': us_daily_increase, 'Spain' : spain_daily_increase, 'France': france_daily_increase, 'Germany': germany_daily_increase,
                          'UK': uk_daily_increase, 'India': india_daily_increase})
Increase_pattern_df.style.background_gradient(cmap='Reds')

# deaths
world_daily_death = daily_increase(total_deaths)
china_daily_death = daily_increase(china_deaths)
italy_daily_death = daily_increase(italy_deaths)
us_daily_death = daily_increase(us_deaths)
spain_daily_death = daily_increase(spain_deaths)
france_daily_death = daily_increase(france_deaths)
germany_daily_death = daily_increase(germany_deaths)
uk_daily_death = daily_increase(uk_deaths)
india_daily_death = daily_increase(india_deaths)

# number of Deaths per day in top affected countries/region
Increase_death_pattern_df = pd.DataFrame({'World Deaths': world_daily_death, 'china': china_daily_death, 'Italy': italy_daily_death,
                          'USA': us_daily_death, 'Spain' : spain_daily_death, 'France': france_daily_death, 'Germany': germany_daily_death,
                          'UK': uk_daily_death, 'India': india_daily_death})
Increase_death_pattern_df.style.background_gradient(cmap='Reds')

# recoveries
china_daily_recovery = daily_increase(china_recoveries1)
italy_daily_recovery = daily_increase(italy_recoveries1)
us_daily_recovery = daily_increase(us_recoveries1)
spain_daily_recovery = daily_increase(spain_recoveries1)
france_daily_recovery = daily_increase(france_recoveries1)
germany_daily_recovery = daily_increase(germany_recoveries1)
uk_daily_recovery = daily_increase(uk_recoveries1)
india_daily_recovery = daily_increase(india_recoveries1)

# number of recoveries per day in top affected countries/region
Increase_recoveries_pattern_df = pd.DataFrame({ 'china': china_daily_recovery, 'Italy': italy_daily_recovery,
                          'USA': us_daily_recovery, 'Spain' : spain_daily_recovery, 'France': france_daily_recovery, 'Germany': germany_daily_recovery,
                          'UK': uk_daily_recovery, 'India': india_daily_recovery})
Increase_recoveries_pattern_df.style.background_gradient(cmap='Reds')
# unique countries
unique_countries =  list(latest_data['Country_Region'].unique())
unique_countries

country_confirmed_cases = []
country_death_cases = [] 
country_active_cases = []
country_recovery_cases = []
country_mortality_rate = [] 

no_cases = []
for i in unique_countries:
    cases = latest_data[latest_data['Country_Region']==i]['Confirmed'].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
# sort countries by the number of confirmed cases
unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_data[latest_data['Country_Region']==unique_countries[i]]['Confirmed'].sum()
    country_death_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Deaths'].sum())
    country_recovery_cases.append(latest_data[latest_data['Country_Region']==unique_countries[i]]['Recovered'].sum())
    country_active_cases.append(country_confirmed_cases[i] - country_death_cases[i] - country_recovery_cases[i])
    country_mortality_rate.append(country_death_cases[i]/country_confirmed_cases[i])

# number of cases per country/region
country_df = pd.DataFrame({'Country Name': unique_countries, 'Number of Confirmed Cases': country_confirmed_cases,
                          'Number of Deaths': country_death_cases, 'Number of Recoveries' : country_recovery_cases, 
                          'Number of Active Cases' : country_active_cases,
                          'Mortality Rate': country_mortality_rate})
country_df.style.background_gradient(cmap='Blues')

unique_provinces =  list(latest_data['Province_State'].unique())
province_confirmed_cases = []
province_country = [] 
province_death_cases = [] 
province_recovery_cases = []
province_mortality_rate = [] 

no_cases = [] 
for i in unique_provinces:
    cases = latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
 
# remove areas with no confirmed cases
for i in no_cases:
    unique_provinces.remove(i)
    
unique_provinces = [k for k, v in sorted(zip(unique_provinces, province_confirmed_cases), key=operator.itemgetter(1), reverse=True)]
for i in range(len(unique_provinces)):
    province_confirmed_cases[i] = latest_data[latest_data['Province_State']==unique_provinces[i]]['Confirmed'].sum()
    province_country.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Country_Region'].unique()[0])
    province_death_cases.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Deaths'].sum())
    province_recovery_cases.append(latest_data[latest_data['Province_State']==unique_provinces[i]]['Recovered'].sum())
    province_mortality_rate.append(province_death_cases[i]/province_confirmed_cases[i])

# number of cases per province/state/city
province_df = pd.DataFrame({'Province/State Name': unique_provinces, 'Country': province_country, 'Number of Confirmed Cases': province_confirmed_cases,
                          'Number of Deaths': province_death_cases, 'Number of Recoveries' : province_recovery_cases,
                          'Mortality Rate': province_mortality_rate})
province_df.style.background_gradient(cmap='Reds')

# Dealing with missing values
nan_indices = [] 

# handle nan if there is any, it is usually a float: float('nan')

for i in range(len(unique_provinces)):
    if type(unique_provinces[i]) == float:
        nan_indices.append(i)

unique_provinces = list(unique_provinces)
province_confirmed_cases = list(province_confirmed_cases)

for i in nan_indices:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)

USA_confirmed = latest_data[latest_data['Country_Region']=='US']['Confirmed'].sum()
outside_USA_confirmed = np.sum(country_confirmed_cases) - USA_confirmed
plt.figure(figsize=(16, 9))
plt.barh('USA', USA_confirmed)
plt.barh('Outside USA', outside_USA_confirmed)
plt.title('Number of Coronavirus Confirmed Cases', size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

print('Outside USA {} cases:'.format(outside_USA_confirmed))
print('USA: {} cases'.format(USA_confirmed))
print('Total: {} cases'.format(USA_confirmed+outside_USA_confirmed))

# Only show 10 countries with the most confirmed cases, the rest are grouped into the other category
visual_unique_countries = [] 
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])

for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)

def plot_bar_graphs(x, y, title):
    plt.figure(figsize=(16, 9))
    plt.barh(x, y)
    plt.title(title, size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
plot_bar_graphs(visual_unique_countries, visual_confirmed_cases, 'Number of Covid-19 Confirmed Cases in Countries/Regions')

def plot_pie_charts(x, y, title):
    c = random.choices(list(mcolors.CSS4_COLORS.values()),k = len(unique_countries))
    plt.figure(figsize=(20,15))
    plt.title(title, size=20)
    plt.pie(y, colors=c)
    plt.legend(x, loc='best', fontsize=15)
    plt.show()
plot_pie_charts(visual_unique_countries, visual_confirmed_cases, 'Covid-19 Confirmed Cases per Country')
# Only show 10 provinces with the most confirmed cases, the rest are grouped into the others category
visual_unique_provinces = [] 
visual_confirmed_cases2 = []
others = np.sum(province_confirmed_cases[10:])
for i in range(len(province_confirmed_cases[:10])):
    visual_unique_provinces.append(unique_provinces[i])
    visual_confirmed_cases2.append(province_confirmed_cases[i])

visual_unique_provinces.append('Others')
visual_confirmed_cases2.append(others)
plot_bar_graphs(visual_unique_provinces, visual_confirmed_cases2, 'Number of Coronavirus Confirmed Cases in Provinces/States')

def plot_pie_country_with_regions(country_name, title):
    regions = list(latest_data[latest_data['Country_Region']==country_name]['Province_State'].unique())
    confirmed_cases = []
    no_cases = [] 
    for i in regions:
        cases = latest_data[latest_data['Province_State']==i]['Confirmed'].sum()
        if cases > 0:
            confirmed_cases.append(cases)
        else:
            no_cases.append(i)

    # remove areas with no confirmed cases
    for i in no_cases:
        regions.remove(i)

    # only show the top 10 states
    regions = [k for k, v in sorted(zip(regions, confirmed_cases), key=operator.itemgetter(1), reverse=True)]

    for i in range(len(regions)):
        confirmed_cases[i] = latest_data[latest_data['Province_State']==regions[i]]['Confirmed'].sum()  
    
    # additional province/state will be considered "others"
    
    if(len(regions)>10):
        regions_10 = regions[:10]
        regions_10.append('Others')
        confirmed_cases_10 = confirmed_cases[:10]
        confirmed_cases_10.append(np.sum(confirmed_cases[10:]))
        plot_pie_charts(regions_10,confirmed_cases_10, title)
    else:
        plot_pie_charts(regions,confirmed_cases, title)

plot_pie_country_with_regions('US', 'COVID-19 Confirmed Cases in the United States')

plot_pie_country_with_regions('France', 'COVID-19 Confirmed Cases in the France')

# preparing data for model building
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)

days_in_future = 20
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forecast[:-20]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.25, shuffle=False) 

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=3)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forecast = poly.fit_transform(future_forecast)
#transforming data for lstm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
lstm_X_train_confirmed = scaler.fit_transform(X_train_confirmed)
lstm_X_test_confirmed = scaler.fit_transform(X_test_confirmed)
lstm_future_forecast = scaler.fit_transform(future_forecast)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
svr_X_train_confirmed = sc.fit_transform(X_train_confirmed)
svr_X_test_confirmed = sc.fit_transform(X_test_confirmed)
svr_future_forecast = sc.fit_transform(future_forecast)


# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])

# lstm
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout, Bidirectional
#from tensorflow.keras.layers import Dense
from keras import callbacks
from keras.layers import Dense, BatchNormalization, Dropout, LSTM
from keras.models import Sequential
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)
# Initialising the LsTM
model = Sequential()
model.add(LSTM(100,return_sequences=True, kernel_initializer='uniform', activation= 'relu',  input_shape=(poly_X_train_confirmed.shape[1],X_test_confirmed.shape[-1])))
model.add(Dropout(0.6))
model.add(LSTM(20,return_sequences=False, kernel_initializer='uniform', activation= 'relu',))
model.add(Dropout(0.6))
model.add(Dense(1))
model.compile(loss='mse', optimizer='rmsprop')
history = model.fit(poly_X_train_confirmed, y_train_confirmed, batch_size = 32, epochs = 100, callbacks=[early_stopping], validation_split=0.2)
lstm_pred = model.predict(poly_X_test_confirmed)
lstm1_pred = model.predict(poly_future_forecast)
print('MAE:', mean_absolute_error(lstm_pred, y_test_confirmed))
print('MSE:',mean_squared_error(lstm_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(lstm_pred)
plt.legend(['Test Data', 'LSTM Prediction'])

# decition tree
# import the regressor
from sklearn.tree import DecisionTreeRegressor   
# create a regressor object
tree = DecisionTreeRegressor(random_state = 0)  
# fit the regressor with X and Y data
tree.fit(svr_X_train_confirmed, y_train_confirmed)
dec_pred1 = tree.predict(svr_X_test_confirmed)
dec_pred2 = tree.predict(svr_future_forecast)
print('MAE:', mean_absolute_error(dec_pred1, y_test_confirmed))
print('MSE:',mean_squared_error(dec_pred1, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(dec_pred1)
plt.legend(['Test Data', 'Decision Tree Prediction'])

#future curves
# cases
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, world_cases)
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# deaths
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_deaths)
plt.title('Number of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#active cases
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_active)
plt.title('Number of Coronavirus Active Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Numner of Active Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# daily insrease worldwide
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_increase)
plt.title('World Daily Increases in Confirmed Cases', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# daily desthd world wide
plt.figure(figsize=(16, 9))
plt.bar(adjusted_dates, world_daily_death)
plt.title('World Daily Increases in Confirmed Deaths', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.plot(future_forecast, pred, linestyle='dashed', color=color)
    plt.title('Number of Coronavirus Cases Over Time', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
plot_predictions(adjusted_dates, world_cases, linear_pred, 'Polynomial Regression Predictions', 'red')
plot_predictions(adjusted_dates, world_cases,lstm1_pred, 'LSTM', 'red')
plot_predictions(adjusted_dates, world_cases, dec_pred2, 'Decision Tree', 'green')

# Future predictions using polynomial regression
linear_pred = linear_pred.reshape(1,-1)[0]
poly_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'Predicted number of Confirmed Cases Worldwide': np.round(linear_pred[-20:])})
poly_df
# lstm prediction
y_pred5 = lstm_pred.reshape(1,-1)[0]
lstm_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'Predicted number of Confirmed Cases Worldwide': np.round(y_pred5[-20:])})
lstm_df
#decition tree prediction
# Future predictions using DT 
svm_df = pd.DataFrame({'Date': future_forecast_dates[-20:], 'DT Predicted # of Confirmed Cases Worldwide': np.round(dec_pred2[-20:])})
svm_df

def country_plot(x, y1, y2, y3, y4, country):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y1)
    plt.title('{} Confirmed Cases'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y2)
    plt.title('{} Daily Increases in Confirmed Cases'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y3)
    plt.title('{} Daily Increases in Deaths'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.bar(x, y4)
    plt.title('{} Daily Increases in Recoveries'.format(country), size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()

country_plot(adjusted_dates, china_cases, china_daily_increase, china_daily_death, china_daily_recovery, 'China')
country_plot(adjusted_dates, italy_cases, italy_daily_increase, italy_daily_death, italy_daily_recovery, 'Italy')
country_plot(adjusted_dates, india_cases, india_daily_increase, india_daily_death, india_daily_recovery, 'India')
country_plot(adjusted_dates, us_cases, us_daily_increase, us_daily_death, us_daily_recovery, 'United States')


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, china_cases)
plt.plot(adjusted_dates, italy_cases)
plt.plot(adjusted_dates, us_cases)
plt.plot(adjusted_dates, spain_cases)
plt.plot(adjusted_dates, france_cases)
plt.plot(adjusted_dates, germany_cases)
plt.plot(adjusted_dates, india_cases)
plt.title('Number of Coronavirus Cases', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['China', 'Italy', 'US', 'Spain', 'France', 'Germany', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, china_deaths)
plt.plot(adjusted_dates, italy_deaths)
plt.plot(adjusted_dates, us_deaths)
plt.plot(adjusted_dates, spain_deaths)
plt.plot(adjusted_dates, france_deaths)
plt.plot(adjusted_dates, germany_deaths)
plt.plot(adjusted_dates, india_deaths)
plt.title('Number of Coronavirus Deaths', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['China', 'Italy', 'US', 'Spain', 'France', 'Germany', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
