import pandas as pd
import graphsUtils as graph

# Import
filename = ".\data\marketing_campaign.csv"
df = pd.read_csv(filename, delimiter = "\t")

description = df.describe()

# Dépenses moyennes par catégories
graph.boxPlotAmounts(df)
graph.meanAmountSpentByAge(df)
graph.meanAmountSpentByIncome(df)
graph.meanAmountSpentByNbKid(df)
graph.meanAmountSpentByNbTeen(df)
graph.meanAmountSpentByEducation(df)
graph.meanAmountSpentByMaritalStatus(df)
graph.meanAmountSpentByDtCustomer(df)