import pandas as pd
from preprocessing.preprocessing import preprocess
from clustering.tuning import tuneAlgorithms
from clustering.algorithmsDictionary import getAlgorithms
from association.utils import getAssociationRules, getRulesforTarget
from utils.utils import analyzeClusters

if __name__ == "__main__":
    # Load dataset
    filename = "../data/marketing_campaign.csv"
    df = pd.read_csv(filename, delimiter = "\t")
    
    # Feature selection
    features = ["Recency", "Income", "TotalExpenses"]
    
    # Preprocessing
    df_total, df = preprocess(df, features)
    
    # Choose algorithms for algorithm tuning
    chosen_algorithms = getAlgorithms(["Kmeans", "GMM", "Spectral"])
    results = tuneAlgorithms(df, chosen_algorithms, components=0.9, verbose = False)
    
    # Get wanted labels
    df_total['Cluster'] = results["GMM"][4]["spherical"]["clusters"]
    analyzeClusters(df_total, features)
    
    # Get association Rules
    list_cols = ["Wine_labeled", "Fruits_labeled", "Meat_labeled", 
                 "Sweets_labeled", "Gold_labeled", "Income_labeled",
                 "Recency_labeled", 
                 #"Cluster", "ChildCount", "Education", "Marital_Status"
                 ]
    test = pd.get_dummies(df_total[list_cols])
    rules = getAssociationRules(df_total[list_cols], min_support = 0.08, max_len = 10)
    
    # Get rules for specific category
    wineRules = getRulesforTarget(rules, "Wine", "Big spender")
    
    print(wineRules)