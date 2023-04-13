import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

adf = pd.read_csv('athletes.csv')
del adf['dob'], adf['id'], adf['name']

adf['nationality']= adf['nationality'].astype(str)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(adf.iloc[:,2:4])
adf.iloc[:, 2:4] = imputer.transform(adf.iloc[:, 2:4])
X = adf.iloc[:, 0:5].values
Y = adf.iloc[:, [5,6,7]].values
X = X.transpose()
le =LabelEncoder()
X[0] = le.fit_transform(X[0])
le = LabelEncoder()
X[4] = le.fit_transform(X[4])
le = LabelEncoder()
X[1] = le.fit_transform(X[1])
gender = adf['sex'].values
nationality =adf['nationality'].values
sports = adf['sport'].values
cat_gen = dict(zip(gender, X[1]))
cat_nat = dict(zip(nationality, X[0]))
cat_spo = dict(zip(sports, X[4]))

cat_data = [cat_nat, cat_gen, cat_spo]
X = X.transpose()
ranForReg = RandomForestRegressor(n_estimators=245, random_state =40, max_depth= 14, min_samples_split=10) 
ranForReg.fit(X, Y)

def Inputs(country, gender, height, weight, sport):
    c = cat_data[0].get(country.upper())
    g = cat_data[1].get(gender.lower())
    s = cat_data[2].get(sport.lower())
    h = height
    w = weight
    list_data = [c,g,h,w,s]
    pred = ranForReg.predict([list_data])
    
    #for Gold medals 
    if pred[0][0] < 0.036:
        gold = 0
    elif pred[0][0] > 0.036 and pred[0][0] < 0.12:
        gold = 1
    elif pred[0][0] > 0.12 and pred[0][0] < 0.9:
        gold = 2
    elif pred[0][0] > 0.9:
        gold= '2+'
    #For Silver medal
    if pred[0][1] < 0.3:
        silver = 0
    elif pred[0][1] > 0.3 and pred[0][1] < 0.5:
        silver = 1
    elif pred[0][1] > 0.5:
        silver = '1+'
    #for Bronze medal
    if pred[0][2] < 0.21:
        bronze = 0
    elif pred[0][2] > 0.21 and pred[0][2] < 4.5:
        bronze = 1
    elif pred[0][2] > 4.5:
        bronze = '1+'

    prediction = f'Total Number of Gold Medals:{gold} \nTotal Number of Silver Medals:{silver}\nTotal Number of Bronze Medals:{bronze}'
    return prediction

nation = input('Country: ') 
gender = input('Gender: ')
height = float(input('Height: '))
weight = float(input('Weight: '))
sports = input('Sports: ')

print(Inputs(nation,gender,height,weight,sports))
