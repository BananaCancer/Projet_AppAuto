# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:07:23 2023

@author: basil
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def getAssociationRules(df_total, list_cols):
    for k in df_total["Cluster"].unique():
        
        current_df = df_total[df_total["Cluster"] == k]
        current_df = current_df[list_cols]
        current_df = pd.get_dummies(current_df, columns = list_cols)
        frequent_items = apriori(current_df, use_colnames=True, min_support=0.08, max_len=10+ 1)
        rules = association_rules(frequent_items, metric='lift', min_threshold=1)
        
        rules = rules.sort_values(['confidence', 'lift'], ascending = [False, False])  
        print(rules.head())  
        
        product='Wine'
        segment='Big spender'
        target = '{\'%s_labeled_%s\'}' %(product,segment)
        results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
        results_personnal_care.head()