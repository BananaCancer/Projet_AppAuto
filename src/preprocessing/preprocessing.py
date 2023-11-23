# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:26:08 2023

@author: basil
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from preprocessing.plotting import plotHist

def columnRename(df):
    column_mapping = {'Year_Birth': 'Birth', 
                      'Kidhome': 'NbKid', 
                      'Teenhome': 'NbTeen',
                      'MntWines': 'Wine', 
                      'MntFruits': 'Fruits',
                      'MntMeatProducts': 'Meat', 
                      'MntSweetProducts': 'Sweets',
                      'MntGoldProds': 'Gold', 
                      'NumDealPurchases': 'NbDealPurchases',
                      'NumWebPurchases': 'NbWebPurchases',
                      'NumCatalogPurchases': 'NbCatalogPurchases', 
                      'NumStorePurchases': 'NbStorePurchases',
                      'NumWebVisitsMonth': 'MonthlyWeb',
                      'AcceptedCmp1': 'Campaign1',
                      'AcceptedCmp2': 'Campaign2',
                      'AcceptedCmp3': 'Campaign3',
                      'AcceptedCmp4': 'Campaign4',
                      'AcceptedCmp5': 'Campaign5',
                      }
    df.rename(columns=column_mapping, inplace=True)

def removeNA(df):
    df.dropna(how='any', inplace = True)

def createNewCols(df):
    df[['JoinDay', 'JoinMonth', 'JoinYear']] = df['Dt_Customer'].str.split('-', expand=True)
    df['JoinDay'] = pd.to_numeric(df['JoinDay'])
    df['JoinMonth'] = pd.to_numeric(df['JoinMonth'])
    df['JoinYear'] = pd.to_numeric(df['JoinYear'])
    df["ChildCount"] = df["NbKid"] + df["NbTeen"]
    df["TotalExpenses"] = df["Wine"] + df["Meat"] + df["Fruits"] + df["Sweets"] + df["Gold"]

    labels = ["Low spender", "Average spender", "High spender", "Big spender"]
    for col in ["Wine", "Fruits", "Meat", "Sweets", "Gold"]:
        df[col + "_labeled"] = pd.qcut(df[col], q=[0, 0.25, 0.5, 0.75, 1], labels=labels)
        
    cut_labels_income = ['Low income', 'Medium income', 'High income', 'Very high income']
    df['Income_labeled'] = pd.qcut(df['Income'], q=4, labels=cut_labels_income)
    
    cut_labels_Seniority = ['New customers', 'Active customers', 'Established customers', 'Loyal customers']
    df['Recency_labeled'] = pd.qcut(df['Recency'], q=4, labels=cut_labels_Seniority)

def deleteOutliers(df, viewHist):
    df.drop(df[df["Birth"] < 1935].index, inplace=True)
    df.drop(df[df["Income"] >= 120000].index, inplace=True)
    if (viewHist):
        plotHist(df)

def encode(df, colnames):
    label_encoder = OrdinalEncoder()
    df[colnames] = label_encoder.fit_transform(df[colnames])

def standardize(df, features):
    scaler = StandardScaler()
    df_total = df.copy()
    features_scaled = [item + "_scaled" for item in features]
    df_total[features_scaled] = scaler.fit_transform(df_total[features])
    df = df_total[features_scaled]
    return df_total, df

def preprocess(df, features, viewHist = True):
    # Rename the columns
    columnRename(df)
    # NA handling
    removeNA(df)
    # New columns creation
    createNewCols(df)
    
    # Delete outliers
    deleteOutliers(df, viewHist)
    
    # Encoding
    encode(df, ["Education", "Marital_Status"])
    
    # Return the scaled data
    return standardize(df, features)