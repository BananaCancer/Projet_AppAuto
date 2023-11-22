import pandas as pd
import graphsUtils as graph
# Import
# Load df
filename = ".\data\marketing_campaign.csv"
df = pd.read_csv(filename, delimiter = "\t")

description = df.describe()
#description.to_csv('description.csv')

graph.boxPlotAmounts(df)