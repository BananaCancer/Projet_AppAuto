from preprocessing.preprocessing import preprocess
from clustering.tuning import tuneAlgorithms
from association.utils import getAssociationRules, getRulesforTarget
from utils.utils import analyzeClusters
from fileManagement.utils import loadFile, fileExists
from fileManagement.model import saveModel, loadModel
from clustering.utils import getClusters
if __name__ == "__main__":
    # Model file
    model_file = "../models/GMM_4_spherical.joblib"
    chosen_algorithms = ["Kmeans", "GMM", "Spectral"]
    
    # Load dataset
    df = loadFile("../data/marketing_campaign.csv", "\t")
    
    # Feature selection
    features = ["Recency", "Income", "TotalExpenses"]
    
    # Preprocessing
    df_total, df = preprocess(df, features)
    
    # Choose algorithms for algorithm tuning
    if fileExists(model_file):
        model = loadModel(model_file)
        df_total['Cluster'] = getClusters(model, df, components=0.9)
    else:
        results = tuneAlgorithms(df, chosen_algorithms, components=0.9, verbose = False)
        df_total['Cluster'] = results["GMM"][4]["spherical"]["clusters"]
        saveModel(results["GMM"][4]["spherical"]["model"], model_file)
        analyzeClusters(df_total, features)
    
    # Get association Rules
    list_cols = ["Wine_labeled", "Fruits_labeled", "Meat_labeled", 
                 "Sweets_labeled", "Gold_labeled", "Income_labeled",
                 "Recency_labeled", "Cluster", "ChildCount", "Education", 
                 "Marital_Status", "Age_group"]
    rules = getAssociationRules(df_total, list_cols, min_support = 0.08, max_len = 10)
    
    # Get rules for specific category
    wineRules = getRulesforTarget(rules, "Wine_labeled_Big spender")
    
    print(wineRules)