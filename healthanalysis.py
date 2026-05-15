import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\alvin\OneDrive\Desktop\health-facilities-gh.csv')
print(df)
print(df.shape)
print(df.describe())
print(df.info())
print(df.head(3))
print(df.tail(3))

print(df['Latitude'])
df = df.drop(columns = ['Longitude' , 'Latitude'])
print(df)

group = df.groupby ('Region')['Ownership'].value_counts()
print(group)

print(df['Region'].unique())

def assign_part(Region):
    if Region == 'Ashanti':
        return 'South'
    elif Region == 'Brong Ahafo':
        return 'Central'
    elif Region == 'Central':
        return 'South'
    elif Region == 'Eastern':
        return 'South-Eastern'
    elif Region == 'Greater Accra':
        return 'South'
    elif Region == 'Northern':
        return 'North-Central'
    elif Region == 'Upper East':
        return 'North-Eastern'
    elif Region == 'Upper West':
        return 'North- Western'
    elif Region == 'Volta':
        return 'South-Eastern'
    elif Region == 'Western':
        return 'Western'
    else:
        return 'Unknown'
df['Part'] = df['Part'] = df['Region'].apply(assign_part)
print(df[['Region' ,'FacilityName' ,'Type' ,'Ownership' ,'Part']])

df = df.drop(columns = ['Town'])
print(df)

print(df[(df['Region'] == 'Northern') | (df['Region'] == 'Upper East')])

print(df.isnull().sum())

df_messy = df.copy()
df_messy.loc[5:10, 'Ownership'] = np.nan
df_messy.loc[15:20, 'Type'] = np.nan

print(df_messy.isnull().sum())

df_messy = df.dropna()
print(df_messy)

df_messy = df.fillna('Unknown')
print(df_messy)

