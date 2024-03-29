# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:07:23 2023

@author: basil
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def getAssociationRules(df_total, list_cols, min_support, max_len):
    df = df_total[list_cols]
    df = pd.get_dummies(df, columns = list_cols)
    
    frequent_items = apriori(df, use_colnames=True, 
                             min_support = min_support, 
                             max_len = max_len + 1)
    rules = association_rules(frequent_items, metric='lift', min_threshold=1)
    return rules

def getRulesforTarget(rules, element):
    results = rules[rules['consequents'].astype(str).str\
                    .contains(element, na=False)]\
        .sort_values(by='confidence', ascending=False)
    return results