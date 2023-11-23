# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:07:23 2023

@author: basil
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def getAssociationRules(df_total, min_support, max_len):
    
    current_df = pd.get_dummies(df_total)
    frequent_items = apriori(current_df, use_colnames=True, 
                             min_support = min_support, 
                             max_len = max_len + 1)
    rules = association_rules(frequent_items, metric='lift', min_threshold=1)
    return rules
    

def getRulesforTarget(rules, product, segment):
    target = '{\'%s_labeled_%s\'}' %(product,segment)
    results = rules[rules['consequents'].astype(str).str\
                    .contains(target, na=False)]\
        .sort_values(by='confidence', ascending=False)
    return results