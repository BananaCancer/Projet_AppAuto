# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:39:49 2023

@author: basil
"""
import matplotlib.pyplot as plt

def plotHist(df):
    for col in df.columns.difference(['ID', "Dt_Customer", "NbKid", "NbTeen", 
                                      "Campaign1", "Campaign2", "Campaign3", 
                                      "Campaign4", "Campaign5", "Response", 
                                      "JoinMonth", "JoinDay", "Complain",
                                      "Z_Revenue", "Z_CostContact"]):    
        plt.hist(df[col], bins='auto', alpha=0.7, rwidth=0.85)
        plt.title(f'Histogram of Values in {col}')
        plt.xlabel('Column Values')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.show()