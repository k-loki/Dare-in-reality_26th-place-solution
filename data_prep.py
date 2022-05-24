import pandas as pd
import numpy as np

train = pd.read_csv(r"Data_DIR_2021\train.csv")

train[' S1'] = train[' S1'].str.split(":").str[0]
train[' S2'] = train[' S2'].str.split(":").str[0]
train[' S3'] = train[' S3'].str.split(":").str[0]
train[' ELAPSED'] = train[' ELAPSED'].str.split(":").str[0]
train[' HOUR'] = train[' HOUR'].str.split(":").str[0]
train['S1_LARGE'] = train['S1_LARGE'].str.split(":").str[0]
train['S2_LARGE'] = train['S2_LARGE'].str.split(":").str[0]
train['S3_LARGE'] = train['S3_LARGE'].str.split(":").str[0]
train['PIT_TIME'] = train['PIT_TIME'].str.split(":").str[0]
train.to_csv("train_v1.csv", index=False)

# Weather data
train_weather = pd.read_csv(r"Data_DIR_2021\train_weather.csv")
train_weather.info()
#split into location due to different number formats
train_weather_l1 = train_weather[train_weather['LOCATION'].isin(['Location 1','Location 2','Location 3','Location 4'])]
train_weather_l1['AIR_TEMP'] = train_weather_l1['AIR_TEMP'] .str.replace(',','.')
train_weather_l1['AIR_TEMP'] = pd.to_numeric(train_weather_l1['AIR_TEMP'])
train_weather_l1['TRACK_TEMP'] = train_weather_l1['TRACK_TEMP'] .str.replace(',','.')
train_weather_l1['TRACK_TEMP'] = pd.to_numeric(train_weather_l1['TRACK_TEMP'])
train_weather_l1['HUMIDITY'] = train_weather_l1['HUMIDITY'] .str.replace(',','.')
train_weather_l1['HUMIDITY'] = pd.to_numeric(train_weather_l1['HUMIDITY'])
train_weather_l1['PRESSURE'] = train_weather_l1['PRESSURE'] .str.replace(',','.')
train_weather_l1['PRESSURE'] = pd.to_numeric(train_weather_l1['PRESSURE'])
train_weather_l1['WIND_SPEED'] = train_weather_l1['WIND_SPEED'] .str.replace(',','.')
train_weather_l1['WIND_SPEED'] = pd.to_numeric(train_weather_l1['WIND_SPEED'])



train_weather_l2 = train_weather[train_weather['LOCATION'].isin(['Location 5','Location 6','Location 7'])]
train_weather_l2['AIR_TEMP'] = train_weather_l2['AIR_TEMP'] .str.replace(',','')
train_weather_l2['AIR_TEMP'] = pd.to_numeric(train_weather_l2['AIR_TEMP'], errors='coerce')
conditions = [
    (train_weather_l2['AIR_TEMP'] > 100)  & (train_weather_l2['AIR_TEMP'] < 1000),
    (train_weather_l2['AIR_TEMP'] > 1000) & (train_weather_l2['AIR_TEMP'] < 10000),
    (train_weather_l2['AIR_TEMP'] > 10000) & (train_weather_l2['AIR_TEMP'] < 100000),
    (train_weather_l2['AIR_TEMP'] > 100000)]
choices = [train_weather_l2['AIR_TEMP']/10,train_weather_l2['AIR_TEMP']/100,
           train_weather_l2['AIR_TEMP']/1000,train_weather_l2['AIR_TEMP']/10000]
train_weather_l2['AIR_TEMP'] = np.select(conditions, choices, default=20)

train_weather_l2['TRACK_TEMP'] = train_weather_l2['TRACK_TEMP'] .str.replace(',','.')
train_weather_l2['TRACK_TEMP'] = pd.to_numeric(train_weather_l2['TRACK_TEMP'], errors='coerce')

train_weather_l2['HUMIDITY'] = train_weather_l2['HUMIDITY'] .str.replace(',','.')
train_weather_l2['HUMIDITY'] = pd.to_numeric(train_weather_l2['HUMIDITY'], errors='coerce')



train_weather_l2['PRESSURE'] = train_weather_l2['PRESSURE'] .str.replace(',','')
train_weather_l2['PRESSURE'] = pd.to_numeric(train_weather_l2['PRESSURE'], errors='coerce')
conditions = [
    (train_weather_l2['PRESSURE'] > 10000) & (train_weather_l2['PRESSURE'] < 20000),
    (train_weather_l2['PRESSURE'] > 20000) & (train_weather_l2['PRESSURE'] < 200000),
    (train_weather_l2['PRESSURE'] > 200000)]

choices = [train_weather_l2['PRESSURE']/10,
           train_weather_l2['PRESSURE']/100,
           train_weather_l2['PRESSURE']/1000]
train_weather_l2['PRESSURE'] = np.select(conditions, choices, default=1000)


train_weather_l2['WIND_SPEED'] = train_weather_l2['WIND_SPEED'] .str.replace(',','')
train_weather_l2['WIND_SPEED'] = pd.to_numeric(train_weather_l2['WIND_SPEED'], errors='coerce')
conditions = [
    (train_weather_l2['WIND_SPEED'] > 10) & (train_weather_l2['WIND_SPEED'] < 100),
    (train_weather_l2['WIND_SPEED'] > 100) & (train_weather_l2['WIND_SPEED'] < 1000),
    (train_weather_l2['WIND_SPEED'] > 1000) & (train_weather_l2['WIND_SPEED'] < 10000),
    (train_weather_l2['WIND_SPEED'] > 10000) & (train_weather_l2['WIND_SPEED'] < 100000),
    (train_weather_l2['WIND_SPEED'] > 100000)]

choices = [train_weather_l2['WIND_SPEED']/10,
           train_weather_l2['WIND_SPEED']/100,
           train_weather_l2['WIND_SPEED']/1000,
           train_weather_l2['WIND_SPEED']/10000,
           train_weather_l2['WIND_SPEED']/100000,
           ]
train_weather_l2['WIND_SPEED'] = np.select(conditions, choices, default=1)

train_weather = pd.concat([train_weather_l2,train_weather_l2])

train_weather['TIME_UTC_STR'] = pd.to_datetime(train_weather['TIME_UTC_STR'])
#train_weather['hour'] = train_weather['TIME_UTC_STR'].dt.hour
#train_weather['date'] = train_weather['TIME_UTC_STR'].dt.day

#train_weather = train_weather.groupby(['LOCATION', 'EVENT', 'hour']).agg({'AIR_TEMP': ['sum', 'max','mean','min'],
#                                                                          'TRACK_TEMP': 'mean',
#                                                                          'HUMIDITY': 'sum'})


train_weather.to_csv("train_weather.csv")
