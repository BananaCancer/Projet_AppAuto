import pandas as pd
import matplotlib.pyplot as plt

def boxPlotAmounts(df):
    colums_to_normalize=['MntWines', 'MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    selected_columns=['MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds']
    df_normalized = df[colums_to_normalize].div(df[colums_to_normalize].sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    df_normalized.boxplot(column=selected_columns,showfliers=False, widths=0.2)
    plt.title('Comparaison des montants dépensés dans chaque catégorie')
    plt.ylabel('Montant en pourcentage du montant total dépensé')
    plt.show()