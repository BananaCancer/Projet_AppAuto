# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:4:09 2023

@author: Diane Lantran
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


#######################################################################
####VISUALISATION DES DEPENSES MOYENNES EN FONCTION DU PROFIL CLIENT###
#######################################################################

def boxPlotAmounts(df):
    #crée le diagramme moustache pour tous les types de dépenses de la base de données
    colums_to_normalize=['MntWines', 'MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    selected_columns=['MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    df_normalized = df[colums_to_normalize].div(df[colums_to_normalize].sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    df_normalized.boxplot(column=selected_columns,showfliers=False, widths=0.2)
    plt.title('Comparaison des montants dépensés dans chaque catégorie')
    plt.ylabel('Montant en pourcentage du montant total dépensé')
    plt.show()

def meanAmountSpentByAge(df):
    #crée le diagramme bar montrant le montant moyen dépensé pour chaque catégorie de dépense et par année de naissance
    grouped_wines = df.groupby('Year_Birth')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Year_Birth')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Year_Birth')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Year_Birth')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Year_Birth')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Year_Birth')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_meat, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_fish, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_gold, on='Year_Birth')

    # Affichage
    width = 0.15
    age_values = merged_df['Year_Birth']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(age_values + i * width, merged_df[f'Mnt{category}'], width=width, label=category)
    plt.xlabel('Année de naissance')
    plt.ylabel('Montant moyen dépensé')
    plt.title('Montant moyen dépensé par catégorie de dépense et par année de naissance')
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByCathegoryAndBirthYear.png')
    plt.show()

def meanAmountSpentByIncome(df):
    # Filter income between 0 and 200000
    df_filtered = df[(df['Income'] >= 0) & (df['Income'] <= 200000)]

    # Create plots showing the average amount spent for each expense category based on income
    grouped_wines = df_filtered.groupby('Income')['MntWines'].mean().reset_index()
    grouped_fruits = df_filtered.groupby('Income')['MntFruits'].mean().reset_index()
    grouped_meat = df_filtered.groupby('Income')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df_filtered.groupby('Income')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df_filtered.groupby('Income')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df_filtered.groupby('Income')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Income')
    merged_df = pd.merge(merged_df, grouped_meat, on='Income')
    merged_df = pd.merge(merged_df, grouped_fish, on='Income')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Income')
    merged_df = pd.merge(merged_df, grouped_gold, on='Income')

    # Display plots
    income_values = merged_df['Income']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.plot(income_values, merged_df[f'Mnt{category}'], label=category)

    plt.xlabel('Revenu annuel')
    plt.ylabel('Montant moyen dépensé')
    plt.title('Montant moyen dépensé par catégorie de dépense et par revenu annuel')
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByCathegoryAndIncome.png')
    plt.show()

def meanAmountSpentByNbKid(df):
    #crée le diagramme bar montrant le montant moyen dépensé pour chaque catégorie de dépense et par nombre d'enfant dans le foyer
    grouped_wines = df.groupby('Kidhome')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Kidhome')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Kidhome')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Kidhome')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Kidhome')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Kidhome')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_meat, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_fish, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_gold, on='Kidhome')

    # Affichage
    width = 0.15
    nbKid_values = merged_df['Kidhome']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(nbKid_values + i * width, merged_df[f'Mnt{category}'], width=width, label=category)
    plt.xlabel("Nombre d'enfants dans le foyer")
    plt.ylabel('Montant moyen dépensé')
    plt.title("Montant moyen dépensé par catégorie de dépense et par nombre d'enfants dans le foyer")
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByNbKid.png')
    plt.show()

def meanAmountSpentByNbTeen(df):
    #crée le diagramme bar montrant le montant moyen dépensé pour chaque catégorie de dépense et par nombre d'adolescent dans le foyer
    grouped_wines = df.groupby('Teenhome')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Teenhome')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Teenhome')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Teenhome')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Teenhome')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Teenhome')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_meat, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_fish, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_gold, on='Teenhome')

    # Affichage
    width = 0.15
    nbTeen_values = merged_df['Teenhome']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(nbTeen_values + i * width, merged_df[f'Mnt{category}'], width=width, label=category)
    plt.xlabel("Nombre d'adolescent dans le foyer")
    plt.ylabel('Montant moyen dépensé')
    plt.title("Montant moyen dépensé par catégorie de dépense et par nombre d'adolescent dans le foyer")
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByNbTeen.png')
    plt.show()

def meanAmountSpentByEducation(df):
    education_mapping = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
    
    # Apply the mapping to the 'Education' column
    df['EducationNumeric'] = df['Education'].map(education_mapping)

    # Group by the numeric Education level
    grouped_wines = df.groupby('EducationNumeric')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('EducationNumeric')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('EducationNumeric')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('EducationNumeric')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('EducationNumeric')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('EducationNumeric')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='EducationNumeric')
    merged_df = pd.merge(merged_df, grouped_meat, on='EducationNumeric')
    merged_df = pd.merge(merged_df, grouped_fish, on='EducationNumeric')
    merged_df = pd.merge(merged_df, grouped_sweet, on='EducationNumeric')
    merged_df = pd.merge(merged_df, grouped_gold, on='EducationNumeric')

    # Define the width of the bars and the offset for each category
    width = 0.15
    offsets = [-2*width, -width, -width/2, width/2, width, 2*width]

    # Plotting
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(merged_df['EducationNumeric'] + offsets[i], merged_df[f'Mnt{category}'], width=width, label=category)

    # Set x-axis ticks to display original Education levels
    plt.xticks(merged_df['EducationNumeric'], df['Education'].unique())

    plt.xlabel("Niveau d'étude")
    plt.ylabel('Montant moyen dépensé')
    plt.title("Montant moyen dépensé par catégorie de dépense et par niveau d'étude")
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByEducation.png')
    plt.show()

def meanAmountSpentByMaritalStatus(df):
    unique_education_labels = df['Marital_Status'].unique()
    print(unique_education_labels)
    marritalStatus_mapping = {'Single': 1, 'Together': 2, 'Married': 3, 'Divorced': 4, 'Widow': 5, 'Alone' : 6, 'Absurd':7, 'YOLO':8}
    
    # Apply the mapping to the 'Education' column
    df['Marital_StatusNumeric'] = df['Marital_Status'].map(marritalStatus_mapping)

    # Group by the numeric Education level
    grouped_wines = df.groupby('Marital_StatusNumeric')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Marital_StatusNumeric')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Marital_StatusNumeric')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Marital_StatusNumeric')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Marital_StatusNumeric')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Marital_StatusNumeric')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Marital_StatusNumeric')
    merged_df = pd.merge(merged_df, grouped_meat, on='Marital_StatusNumeric')
    merged_df = pd.merge(merged_df, grouped_fish, on='Marital_StatusNumeric')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Marital_StatusNumeric')
    merged_df = pd.merge(merged_df, grouped_gold, on='Marital_StatusNumeric')

    # Define the width of the bars and the offset for each category
    width = 0.15
    offsets = [-4*width, -3*width, -2*width, -width, width, 2*width, 3*width, 4*width] 

    # Plotting
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(merged_df['Marital_StatusNumeric'] + offsets[i], merged_df[f'Mnt{category}'], width=width, label=category)

    # Set x-axis ticks to display original Marrital Status levels
    plt.xticks(merged_df['Marital_StatusNumeric'], df['Marital_Status'].unique())

    plt.xlabel("Statut marrital")
    plt.ylabel('Montant moyen dépensé')
    plt.title("Montant moyen dépensé par catégorie de dépense et par statut marrital")
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByMarritalStatus.png')
    plt.show()

def meanAmountSpentByDtCustomer(df):
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

    # Group by 'Dt_Customer'
    grouped_wines = df.groupby('Dt_Customer')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Dt_Customer')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Dt_Customer')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Dt_Customer')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Dt_Customer')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Dt_Customer')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_meat, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_fish, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_gold, on='Dt_Customer')

    # Sort by 'Dt_Customer' for better visualization
    merged_df = merged_df.sort_values(by='Dt_Customer')

    # Plotting
    date_values = merged_df['Dt_Customer']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    
    for category in categories:
        plt.plot(date_values, merged_df[f'Mnt{category}'], label=category)

    plt.xlabel('Date de souscription')
    plt.ylabel('Montant moyen dépensé')
    plt.title('Montant moyen dépensé par catégorie de dépense et par date de souscription')
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByCategoryAndDate.png')
    plt.show()

def meanAmountSpentByRecency(df):
    #crée le diagramme bar montrant le montant moyen dépensé pour chaque catégorie de dépense et par nombre d'adolescent dans le foyer
    grouped_wines = df.groupby('Recency')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Recency')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Recency')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Recency')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Recency')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Recency')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Recency')
    merged_df = pd.merge(merged_df, grouped_meat, on='Recency')
    merged_df = pd.merge(merged_df, grouped_fish, on='Recency')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Recency')
    merged_df = pd.merge(merged_df, grouped_gold, on='Recency')

    # Affichage
    width = 0.15
    nbTeen_values = merged_df['Recency']
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    for i, category in enumerate(categories):
        plt.bar(nbTeen_values + i * width, merged_df[f'Mnt{category}'], width=width, label=category)
    plt.xlabel("Frequence d'achat")
    plt.ylabel('Montant moyen dépensé')
    plt.title("Montant moyen dépensé par catégorie de dépense et par frequence d'achat")
    plt.legend()
    plt.savefig('graphs_BDD/meanAmountByRecency.png')
    plt.show()

##########################################
###CALCUL DU COEFFICIENT DE CORRELATION###
##########################################

def meanAmountBirthYScatter(df):
    #scatter le montant moyen dépensé pour chaque catégorie de dépense en fonction de l'année de naissance du client
    grouped_wines = df.groupby('Year_Birth')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Year_Birth')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Year_Birth')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Year_Birth')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Year_Birth')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Year_Birth')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_meat, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_fish, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Year_Birth')
    merged_df = pd.merge(merged_df, grouped_gold, on='Year_Birth')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        plt.scatter(merged_df['Year_Birth'], merged_df[f'Mnt{category}'], label=category)
    plt.title('Depenses moyennes par catégorie et par année de naissance')
    plt.xlabel("Année de naissance")
    plt.ylabel("Dépenses moyenne par catégorie")
    plt.xticks(rotation=45, ha='right')
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(merged_df['Year_Birth'], merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig('graphs_BDD/correlationBirthYearExpenses.png')
    plt.show()

def meanAmountIncome(df):
    # Scatter plot of the average amount spent for each expense category based on income
    grouped_wines = df.groupby('Income')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Income')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Income')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Income')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Income')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Income')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Income')
    merged_df = pd.merge(merged_df, grouped_meat, on='Income')
    merged_df = pd.merge(merged_df, grouped_fish, on='Income')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Income')
    merged_df = pd.merge(merged_df, grouped_gold, on='Income')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(categories):
        plt.scatter(merged_df['Income'], merged_df[f'Mnt{category}'], label=category)
        plt.plot(merged_df['Income'], merged_df[f'Mnt{category}'], linestyle='dashed', marker='o')

    plt.title('Dépenses moyennes par catégorie et par revenu annuel')
    plt.xlabel("Revenu annuel")
    plt.ylabel("Dépenses moyennes par catégorie")
    plt.xticks(rotation=45, ha='right')
    
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(merged_df['Income'], merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationIncomExpenses.png')
    plt.show()

def meanAmountNbKid(df):
    # Scatter plot of the average amount spent for each expense category based on the number of kids
    grouped_wines = df.groupby('Kidhome')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Kidhome')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Kidhome')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Kidhome')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Kidhome')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Kidhome')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_meat, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_fish, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Kidhome')
    merged_df = pd.merge(merged_df, grouped_gold, on='Kidhome')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(categories):
        plt.scatter(merged_df['Kidhome'], merged_df[f'Mnt{category}'], label=category)
        plt.plot(merged_df['Kidhome'], merged_df[f'Mnt{category}'], linestyle='dashed', marker='o')

    plt.title("Dépenses moyennes par nombre d'enfants dans le foyer")
    plt.xlabel("Nombre d'enfants")
    plt.ylabel("Dépenses moyennes par catégorie")
    
    # Set integer ticks on the x-axis
    plt.xticks(list(map(int, merged_df['Kidhome'])))
    
    plt.xticks(rotation=45, ha='right')
    
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(merged_df['Kidhome'], merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationNbKidExpenses.png')
    plt.show()

def meanAmountNbTeen(df):
    # Scatter plot of the average amount spent for each expense category based on the number of teenagers
    grouped_wines = df.groupby('Teenhome')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Teenhome')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Teenhome')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Teenhome')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Teenhome')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Teenhome')['MntGoldProds'].mean().reset_index()
    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_meat, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_fish, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Teenhome')
    merged_df = pd.merge(merged_df, grouped_gold, on='Teenhome')
    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(categories):
        plt.scatter(merged_df['Teenhome'], merged_df[f'Mnt{category}'], label=category)
        plt.plot(merged_df['Teenhome'], merged_df[f'Mnt{category}'], linestyle='dashed', marker='o')

    plt.title("Dépenses moyennes par nombre d'adolescents dans le foyer")
    plt.xlabel("Nombre d'adolescents")
    plt.ylabel("Dépenses moyennes par catégorie")
    
    # Set integer ticks on the x-axis
    plt.xticks(list(map(int, merged_df['Teenhome'])))
    plt.xticks(rotation=45, ha='right')
    
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(merged_df['Teenhome'], merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationNbTeenExpenses.png')
    plt.show()

def meanAmountEducation(df):
    # Scatter plot of the average amount spent for each expense category based on the level of education
    grouped_wines = df.groupby('Education')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Education')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Education')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Education')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Education')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Education')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Education')
    merged_df = pd.merge(merged_df, grouped_meat, on='Education')
    merged_df = pd.merge(merged_df, grouped_fish, on='Education')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Education')
    merged_df = pd.merge(merged_df, grouped_gold, on='Education')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(categories):
        plt.scatter(range(len(merged_df['Education'])), merged_df[f'Mnt{category}'], label=category)
        plt.plot(range(len(merged_df['Education'])), merged_df[f'Mnt{category}'], linestyle='dashed', marker='o')

    plt.title("Dépenses moyennes par niveau d'éducation")
    plt.xlabel("Niveau d'éducation")
    plt.ylabel("Dépenses moyennes par catégorie")
    
    # Set x-axis ticks based on the index of unique values
    plt.xticks(range(len(merged_df['Education'])), merged_df['Education'], rotation=45, ha='right')
    
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(range(len(merged_df['Education'])), merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationEducationExpenses.png')
    plt.show()

def meanAmountMaritalStatus(df):
    # Scatter plot of the average amount spent for each expense category based on marital status
    grouped_wines = df.groupby('Marital_Status')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Marital_Status')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Marital_Status')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Marital_Status')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Marital_Status')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Marital_Status')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Marital_Status')
    merged_df = pd.merge(merged_df, grouped_meat, on='Marital_Status')
    merged_df = pd.merge(merged_df, grouped_fish, on='Marital_Status')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Marital_Status')
    merged_df = pd.merge(merged_df, grouped_gold, on='Marital_Status')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    
    for i, category in enumerate(categories):
        plt.scatter(range(len(merged_df['Marital_Status'])), merged_df[f'Mnt{category}'], label=category)
        plt.plot(range(len(merged_df['Marital_Status'])), merged_df[f'Mnt{category}'], linestyle='dashed', marker='o')

    plt.title("Dépenses moyennes par statut marital")
    plt.xlabel("Statut marital")
    plt.ylabel("Dépenses moyennes par catégorie")
    
    # Set x-axis ticks based on the index of unique values
    plt.xticks(range(len(merged_df['Marital_Status'])), merged_df['Marital_Status'], rotation=45, ha='right')
    
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(range(len(merged_df['Marital_Status'])), merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationMarital_StatusExpenses.png')
    plt.show()

def meanAmountDtCustomer(df):
    # Convert 'Dt_Customer' column to datetime objects
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

    # Scatter plot of the average amount spent for each expense category based on 'Dt_Customer'
    grouped_wines = df.groupby('Dt_Customer')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Dt_Customer')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Dt_Customer')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Dt_Customer')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Dt_Customer')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Dt_Customer')['MntGoldProds'].mean().reset_index()

    # Merge the data
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_meat, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_fish, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Dt_Customer')
    merged_df = pd.merge(merged_df, grouped_gold, on='Dt_Customer')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        # Convert datetime to Unix timestamp
        timestamp = merged_df['Dt_Customer'].astype('int64') // 10**9  # Convert nanoseconds to seconds
        plt.scatter(timestamp, merged_df[f'Mnt{category}'], label=category)

    plt.title("Dépenses moyennes par date")
    plt.xlabel("Date d'inscription")
    plt.ylabel("Dépenses moyennes par catégorie")
    plt.xticks(rotation=45, ha='right')

    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(timestamp, merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)

    plt.savefig('graphs_BDD/correlationDt_CustomerExpenses.png')
    plt.show()


def meanAmountRecency(df):
    #scatter le montant moyen dépensé pour chaque catégorie de dépense en fonction de la fréquence d'achat
    grouped_wines = df.groupby('Recency')['MntWines'].mean().reset_index()
    grouped_fruits = df.groupby('Recency')['MntFruits'].mean().reset_index()
    grouped_meat = df.groupby('Recency')['MntMeatProducts'].mean().reset_index()
    grouped_fish = df.groupby('Recency')['MntFishProducts'].mean().reset_index()
    grouped_sweet = df.groupby('Recency')['MntSweetProducts'].mean().reset_index()
    grouped_gold = df.groupby('Recency')['MntGoldProds'].mean().reset_index()

    # Fusionne les données
    merged_df = pd.merge(grouped_wines, grouped_fruits, on='Recency')
    merged_df = pd.merge(merged_df, grouped_meat, on='Recency')
    merged_df = pd.merge(merged_df, grouped_fish, on='Recency')
    merged_df = pd.merge(merged_df, grouped_sweet, on='Recency')
    merged_df = pd.merge(merged_df, grouped_gold, on='Recency')

    categories = ['Wines', 'Fruits', 'MeatProducts', 'FishProducts', 'SweetProducts', 'GoldProds']
    plt.figure(figsize=(12, 8))
    for i, category in enumerate(categories):
        plt.scatter(merged_df['Recency'], merged_df[f'Mnt{category}'], label=category)
    plt.title("Depenses moyennes par catégorie et par frequence d'achat")
    plt.xlabel("Fréquence d'achat")
    plt.ylabel("Dépenses moyenne par catégorie")
    plt.xticks(rotation=45, ha='right')
    for i, category in enumerate(categories):
        correlation_coefficient, _ = pearsonr(merged_df['Recency'], merged_df[f'Mnt{category}'])
        plt.text(0.8, 0.9 - 0.1*i, f"{category} - Correlation Coefficient: {correlation_coefficient:.2f}",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
    plt.savefig('graphs_BDD/correlationRecencyExpenses.png')
    plt.show()