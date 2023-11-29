import pandas as pd
import graphsUtils as graph

# Import
filename = ".\data\marketing_campaign.csv"
df = pd.read_csv(filename, delimiter = "\t")

description = df.describe()

# DEPENSE MOYENNES PAR CATEGORIE
graph.boxPlotAmounts(df) 
graph.meanAmountSpentByAge(df) #motif parabolique pour le vin et la viande avec des plus faibles dépenses pour les personnes nées au début des années 80
graph.meanAmountSpentByIncome(df) #a mesure que le revenu croit les depenses en viande/vin augmentent relation visiblement lineaire
graph.meanAmountSpentByNbKid(df) #des que des enfants sont présents dans le foyer les montants consacrés au vin chutent
graph.meanAmountSpentByNbTeen(df) #pas de corrélation visible
graph.meanAmountSpentByEducation(df) #relation linéaire visible entre produits de luxe (vin/viande) et niveau d'étude
graph.meanAmountSpentByMaritalStatus(df) #pas de corrélation visible
graph.meanAmountSpentByDtCustomer(df) #pics de dépense réguliers observés sur les débuts/fins de mois
graph.meanAmountSpentByRecency(df) #pas de correlation apparente

# MESURE DE LA CORRELATION
graph.meanAmountBirthYScatter(df)
graph.meanAmountIncome(df)
graph.meanAmountNbKid(df)
graph.meanAmountNbTeen(df)
graph.meanAmountEducation(df)
graph.meanAmountMaritalStatus(df)
graph.meanAmountDtCustomer(df)
graph.meanAmountRecency(df)