import pandas as pd
from preprocessing.preprocessing import preprocess
from clustering.tuning import tuneAlgorithms
from clustering.algorithmsDictionary import getAlgorithms
from association.utils import getAssociationRules
from utils.utils import findCommonPoints

if __name__ == "__main__":
    # Load dataset
    filename = "../data/marketing_campaign.csv"
    df = pd.read_csv(filename, delimiter = "\t")
    
    # Feature selection
    #features = ["Birth", "Marital_Status", "Education"]
    features = ["Recency", "Income", "TotalExpenses"]
    
    # Preprocessing
    df_total, df = preprocess(df, features)
    
    # Choose algorithms for algorithm tuning
    chosen_algorithms = getAlgorithms(["Kmeans", "GMM", "Spectral"])
    results = tuneAlgorithms(df, chosen_algorithms, components=0.9, verbose = False)
    
    #Get wanted labels
    df_total['Cluster'] = results["GMM"][4]["spherical"]["clusters"]
    findCommonPoints(df_total, features)
    
    list_cols = ["Wine_labeled", "Fruits_labeled", "Meat_labeled", 
                 "Sweets_labeled", "Gold_labeled"]
    getAssociationRules(df_total, list_cols)
    