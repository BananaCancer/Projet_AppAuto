# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:11:36 2023

@author: basil
"""
def analyzeClusters(df, features):
    for cluster in df['Cluster'].unique():
        chosen_data = df[df["Cluster"] == cluster]
        if (len(chosen_data) >= 10 and len(chosen_data) < 2200):
            print(f"Cluster n°{cluster}: {len(chosen_data)} indivs")
            for feature in features:
                if feature in ["NbTeen", "NbKid", "Marital_Status", "Education"]:
                    total_count = len(df[feature].unique())
                    local_count = len(chosen_data[feature].unique())
                    print(f"{feature}: {local_count}/{total_count}")
                elif feature in ["Recency", "Income", "Birth", "Wine", "Fruits", "Meat", "MntFishProducts", "Sweets", "Gold", ]:
                    total_min= df[feature].min()
                    total_max = df[feature].max()
                    local_min = chosen_data[feature].min()
                    local_max = chosen_data[feature].max()
                    print(f"{feature} range: {local_min}-{local_max} ({total_min}-{total_max})")        
            
            for col in ["Recency", "Income", "Wine", "Fruits", "Meat", "MntFishProducts", "Sweets", "Gold", ]:
                total_min= df[col].min()
                total_max = df[col].max()
                local_min = chosen_data[col].min()
                local_max = chosen_data[col].max()
                print(f"{col} range: {local_min}-{local_max} ({total_min}-{total_max})")        
        print("\n")