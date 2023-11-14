import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import itertools
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def findCommonPoints(df, features):
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
            """
            for col in ["NbTeen", "NbKid", "Marital_Status", "Education"]:
                total_count = len(df[col].unique())
                local_count = len(chosen_data[col].unique())
                print(f"{col}: {local_count}/{total_count}")
            """
            
            for col in ["Recency", "Income", "Wine", "Fruits", "Meat", "MntFishProducts", "Sweets", "Gold", ]:
                total_min= df[col].min()
                total_max = df[col].max()
                local_min = chosen_data[col].min()
                local_max = chosen_data[col].max()
                print(f"{col} range: {local_min}-{local_max} ({total_min}-{total_max})")        
        print("\n")

def plotHist(df):
    for col in df.columns.difference(['ID', "Dt_Customer", "NbKid", "NbTeen", 
                                      "Campaign1", "Campaign2", "Campaign3", 
                                      "Campaign4", "Campaign5", "Response", 
                                      "JoinMonth", "JoinDay", "Complain",
                                      "Z_Revenue", "Z_CostContact"]):    
        plt.hist(df[col], bins='auto', alpha=0.7, rwidth=0.85)
        plt.title(f'Histogram of Values in {col}')
        plt.xlabel('Column Values')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.show()

# Load df
filename = "marketing_campaign.csv"
df = pd.read_csv(filename, delimiter = "\t")

# Choose columns for features
# Old attempts: Bad cause too many cols
# features = ['Birth', "Education", "Marital_Status", "Income", "NbKid", 
#            "NbTeen", "Dt_Customer", "Recency"]

# features = ['Wine', "Fruits", "Meat", "Sweets", "MntFishProducts"]
# features = df.columns.difference(['ID'])
# features = ["Recency", 'Wine', "Fruits", "Meat", "Sweets", "MntFishProducts", ]

# New attempts
features = ["Birth", "Marital_Status", "Education"]
# features = ["Recency", "Income"]

# Columns renaming
column_mapping = {'Year_Birth': 'Birth', 
                  'Kidhome': 'NbKid', 
                  'Teenhome': 'NbTeen',
                  'MntWines': 'Wine', 
                  'MntFruits': 'Fruits',
                  'MntMeatProducts': 'Meat', 
                  'MntSweetProducts': 'Sweets',
                  'MntGoldProds': 'Gold', 
                  'NumDealPurchases': 'NbDealPurchases',
                  'NumWebPurchases': 'NbWebPurchases',
                  'NumCatalogPurchases': 'NbCatalogPurchases', 
                  'NumStorePurchases': 'NbStorePurchases',
                  'NumWebVisitsMonth': 'MonthlyWeb',
                  'AcceptedCmp1': 'Campaign1',
                  'AcceptedCmp2': 'Campaign2',
                  'AcceptedCmp3': 'Campaign3',
                  'AcceptedCmp4': 'Campaign4',
                  'AcceptedCmp5': 'Campaign5',
                  }
df.rename(columns=column_mapping, inplace=True)

# NA handling
df = df.dropna(how='any')

# New columns creation
df[['JoinDay', 'JoinMonth', 'JoinYear']] = df['Dt_Customer'].str.split('-', expand=True)
df['JoinDay'] = pd.to_numeric(df['JoinDay'])
df['JoinMonth'] = pd.to_numeric(df['JoinMonth'])
df['JoinYear'] = pd.to_numeric(df['JoinYear'])
df["ChildCount"] = df["NbKid"] + df["NbTeen"]
df["Total"] = df["Wine"] + df["Meat"] + df["Fruits"] + df["Sweets"] + df["Gold"]

for col in ["Wine", "Fruits", "Meat", "Sweets", "Gold"]:
    labels = ["Low spender", "Average spender", "High spender", "Big spender"]
    df[col + "_labeled"] = pd.qcut(df[col], q=[0, 0.25, 0.5, 0.75, 1], labels=labels)


# Delete outliers
df = df[df["Birth"] >= 1935]
df = df[df["Income"] < 120000] # Suppression de 32 obs

# View histograms
plotHist(df)

# Encoding
list_cols_to_encode = ["Education", "Marital_Status"]
label_encoder = OrdinalEncoder()
df[list_cols_to_encode] = label_encoder.fit_transform(df[list_cols_to_encode])

# Standardize the data (scaling)
scaler = StandardScaler()
df_total = df.copy()
features_scaled = [item + "_scaled" for item in features]
df_total[features_scaled] = scaler.fit_transform(df_total[features])
df = df_total[features_scaled]

test = df_total.describe()


# Use the Elbow Method to find the optimal number of clusters (K)
if True:
    sse = []
    for k in range(4, 5):  # Try different K values
        print(k)
        kmeans = KMeans(n_clusters=k, n_init = 5, random_state=42)
        clusters = kmeans.fit_predict(df)
        df_total['Cluster'] = clusters
        findCommonPoints(df_total, features)
        print("----------------------\n")
        # transactions = df_total.groupby('Cluster')['Item'].apply(list).tolist()
        df_apriori = df_total[["Wine_labeled", "Fruits_labeled", "Meat_labeled", 
                         "Sweets_labeled", "Gold_labeled"]]
        frequent_items = apriori(df_apriori, use_colnames=True, min_support=0.08, max_len=10+ 1)
        rules = association_rules(df_apriori, metric='lift', min_threshold=1)
        
        product='Wines'
        segment='Big spender'
        target = '{\'%s_segment_%s\'}' %(product,segment)
        results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
        results_personnal_care.head()
        """
        sse.append(kmeans.inertia_)
        
        for cluster in df_total['Cluster'].unique():
            chosen_data = df_total[df_total["Cluster"] == cluster]
            count = chosen_data.shape[0]
            columns_with_same_values = chosen_data.columns[chosen_data.apply(lambda x: x.nunique() == 1)].tolist()
            # Nice k: 10
            print(f"For {k}: cols with same vals: {columns_with_same_values} (size: {count})")
        """
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 30), sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.show()
    """


if False:
    epslist = [5, 6, 7, 8, 9, 10, 15] # [9]
    min_sampleslist = [2, 3, 4, 5, 6, 7, 8, 9]# [4]
    all_triplets = list(itertools.product(epslist, min_sampleslist))
    for eps, min_samples in all_triplets:
        print(f"Current parameters: {eps}, {min_samples}")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(df)
        silhouette_avg = silhouette_score(df, clustering)
        df_total['Cluster'] = clustering
        findCommonPoints(df_total, features)
        print("\n")
        """
        print(len(df_total["Cluster"].unique()))
        print(f"Silhouette Score for {eps, min_samples}: {silhouette_avg}")
        # (9, 9), 
        for cluster in df_total['Cluster'].unique():
            chosen_data = df_total[df_total["Cluster"] == cluster]
            count = chosen_data.shape[0]
            columns_with_same_values = chosen_data.columns[chosen_data.apply(lambda x: x.nunique() == 1)].tolist()
            print(f"For {cluster}: cols with same vals: {columns_with_same_values} (size: {count})")
        """

if False:
    n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9] # [9]
    metric_list = ["euclidean"]#["euclidean", "manhattan", "l1", "l2", "cosine"]# [4]
    linkage_list = ["ward", "complete", "average", "single"]
    all_triplets = list(itertools.product(n_clusters_list, metric_list, linkage_list))
    for n, metric, linkage in all_triplets:
        print(f"Current parameters: {n}, {metric}, {linkage}")
        clustering = AgglomerativeClustering(n_clusters = n, metric = metric, linkage = linkage).fit_predict(df)
        df_total['Cluster'] = clustering
        # (9, 9), 
        """
        for cluster in df_total['Cluster'].unique():
            chosen_data = df_total[df_total["Cluster"] == cluster]
            if (len(chosen_data) >= 10):
                for col in df_total.columns:
                    print(f"Cluster n°{cluster}, ({len(chosen_data)} indivs): Number of different values for {col}:{len(chosen_data[col].unique())}/{len(df_total[col].unique())}")
                count = chosen_data.shape[0]
                columns_with_same_values = chosen_data.columns[chosen_data.apply(lambda x: x.nunique() == 1)].tolist()
            #print(f"For {cluster}/{df['Cluster'].unique()}: cols with same vals: {columns_with_same_values} (size: {count})")
        """
        findCommonPoints(df_total, features)
        print("\n")
        
