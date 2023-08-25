# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Importing the dataset
covidatabase = pd.read_csv("covid-19-country-statistics-dataset-mod3.csv")
covidatabase.corr()


valoresPerdidos = covidatabase.isnull().sum()
totalCelulas = np.product(covidatabase.shape)
totalPerdidos = valoresPerdidos.sum()
percPerdidos = (totalPerdidos/totalCelulas)*100
percPerdidos

covidatabaseDrop = covidatabase.dropna()
covidatabaseDrop

covidatabase.corr(method="pearson")

covidatabase["Total Cases"] = covidatabase["Total Cases"].astype(np.float64)

training = covidatabase[["Total Cases", "Total Recovered"]].drop([0,5], axis=0).dropna().copy()
training

imputer = IterativeImputer(random_state=0)
imputer.fit(training)
iterativetr = pd.DataFrame(imputer.transform(covidatabase[["Total Cases", "Total Recovered"]])).abs()
iterativetr.corr(method="pearson")

#https://scikit-learn.org/stable/modules/preprocessing.html#normalization
imputer = KNNImputer(n_neighbors=15, weights="uniform")
imputer.fit(training)
trimputed = imputer.transform(covidatabase[["Total Cases", "Total Recovered"]])
trimputed = pd.DataFrame(trimputed)
trimputed.corr(method="pearson")

covidatabase["Total Recovered"] = trimputed.iloc[:,1].abs()
covidatabase

training1 = covidatabase[["Total Deaths", "Total Recovered"]].dropna().copy()
training1

imputer = IterativeImputer(random_state=0)
imputer.fit(training1)
resultadotd = imputer.transform(covidatabase[["Total Deaths","Total Recovered"]])
resultadotd = pd.DataFrame(resultadotd)
resultadotd.corr(method="pearson")

imputer = KNNImputer(n_neighbors=15, weights="uniform")
imputer.fit(training1)
tr = imputer.transform(covidatabase[["Total Deaths","Total Recovered"]])
tr = pd.DataFrame(tr)
tr.corr(method="pearson")

covidatabase["Total Deaths"] = tr.iloc[:,0]
covidatabase

covidatabase.corr()

training2 = covidatabase[["Total Deaths", "Serious_Critical"]].drop([0,5], axis=0).dropna().copy()
training2

imputer = IterativeImputer(random_state=0)
imputer.fit(training2)
resultadotd1 = imputer.transform(covidatabase[["Total Deaths","Serious_Critical"]])
resultadotd1 = pd.DataFrame(resultadotd1)
resultadotd1.corr(method="pearson")

imputer = KNNImputer(n_neighbors=15, weights="uniform")
imputer.fit(training2)
tr1 = imputer.transform(covidatabase[["Total Deaths","Serious_Critical"]])
tr1 = pd.DataFrame(tr1)
tr1.corr(method="pearson")

covidatabase["Serious_Critical"] = tr1.iloc[:,1]
covidatabase

covidatabase.corr()

training4 = covidatabase[["Total Recovered", "Total Tests"]].drop([0,5], axis=0).dropna().copy()
training4

imputer = IterativeImputer(random_state=0)
imputer.fit(training4)
resultadotd2 = imputer.transform(covidatabase[["Total Recovered", "Total Tests"]])
resultadotd2 = pd.DataFrame(resultadotd2)
resultadotd2.corr(method="pearson")

imputer = KNNImputer(n_neighbors=15, weights="uniform")
imputer.fit(training4)
tr2 = imputer.transform(covidatabase[["Total Recovered", "Total Tests"]])
tr2 = pd.DataFrame(tr2)
tr2.corr(method="pearson")

covidatabase["Total Tests"] = tr2.iloc[:,1]
covidatabase

covidatabase.corr()