import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy import stats
import os

# Set environment variable to prevent memory leak warning
os.environ['OMP_NUM_THREADS'] = '2'

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic customer data
def generate_customer_data(num_customers=300):
    """
    Generate synthetic customer data with:
    - Customer ID
    - Age
    - Annual Income (k$)
    - Spending Score (1-100)
    - Online Purchase Frequency (times/month)
    """
    # Generate base data
    ages = np.random.randint(18, 70, size=num_customers)
    incomes = np.random.normal(50, 15, num_customers).clip(20, 120)
    spending_scores = np.random.randint(1, 101, size=num_customers)
    online_freq = np.random.poisson(5, size=num_customers).clip(1, 20)
    
    # Create some meaningful clusters artificially
    # Cluster 1: Young high spenders
    mask1 = (ages < 30) & (incomes > 60)
    spending_scores[mask1] = np.random.randint(70, 96, size=mask1.sum())
    
    # Cluster 2: Middle-aged savers
    mask2 = (ages >= 30) & (ages < 50) & (incomes < 60)
    spending_scores[mask2] = np.random.randint(20, 50, size=mask2.sum())
    
    # Cluster 3: Retired with medium spending
    mask3 = (ages >= 60)
    spending_scores[mask3] = np.random.randint(40, 70, size=mask3.sum())
    
    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': range(1, num_customers + 1),
        'Age': ages,
        'AnnualIncome': incomes,
        'SpendingScore': spending_scores,
        'OnlineFrequency': online_freq
    })
    
    return data

# Step 2: Data Exploration and Cleaning
def explore_and_clean_data(df):
    """Explore and clean the customer data"""
    print("\n=== Data Overview ===")
    print(df.head())
    
    print("\n=== Data Information ===")
    print(df.info())
    
    print("\n=== Descriptive Statistics ===")
    print(df.describe())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Check for outliers using z-score
    numeric_cols = ['Age', 'AnnualIncome', 'SpendingScore', 'OnlineFrequency']
    z_scores = stats.zscore(df[numeric_cols])
    outliers = (np.abs(z_scores) > 3).any(axis=1)
    print(f"\nNumber of outliers detected: {outliers.sum()}")
    
    # Remove outliers
    df_clean = df[~outliers].copy()
    print(f"Data shape after removing outliers: {df_clean.shape}")
    
    return df_clean

# Step 3: Feature Engineering and Scaling
def prepare_features(df):
    """Prepare features for clustering"""
    # Select relevant features
    features = df[['AnnualIncome', 'SpendingScore', 'OnlineFrequency']]
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, features

# Step 4: Determine Optimal Number of Clusters
def find_optimal_clusters(data, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score"""
    wcss = []  # Within-Cluster Sum of Square
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicitly set n_init
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    # Plot elbow curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.show()
    
    return wcss, silhouette_scores

# Step 5: Perform Clustering
def perform_clustering(data, n_clusters=4):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    
    return clusters, kmeans

# Step 6: Analyze and Visualize Results
def analyze_and_visualize(df, clusters, features):
    """Analyze and visualize the clustering results"""
    # Add cluster labels to original data
    df['Cluster'] = clusters
    
    # Cluster statistics
    print("\n=== Cluster Statistics ===")
    cluster_stats = df.groupby('Cluster')[['Age', 'AnnualIncome', 'SpendingScore', 'OnlineFrequency']].mean()
    print(cluster_stats)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Income vs Spending Score
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(features[:, 0], features[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title('Income vs Spending Score')
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.colorbar(scatter, label='Cluster')
    
    # Plot 2: Age Distribution by Cluster
    plt.subplot(2, 2, 2)
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        plt.hist(cluster_data['Age'], bins=20, alpha=0.5, label=f'Cluster {cluster}')
    plt.title('Age Distribution by Cluster')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot 3: Online Frequency vs Spending Score
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(features[:, 2], features[:, 1], c=clusters, cmap='plasma', alpha=0.7)
    plt.title('Online Frequency vs Spending Score')
    plt.xlabel('Online Frequency (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.colorbar(scatter, label='Cluster')
    
    # Plot 4: Cluster Sizes
    plt.subplot(2, 2, 4)
    cluster_counts = df['Cluster'].value_counts().sort_index()
    plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Customers')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Main execution
if __name__ == "__main__":
    # Step 1: Generate data
    print("Generating customer data...")
    customer_data = generate_customer_data(500)
    
    # Step 2: Explore and clean data
    print("\nExploring and cleaning data...")
    clean_data = explore_and_clean_data(customer_data)
    
    # Step 3: Prepare features
    print("\nPreparing features for clustering...")
    scaled_features, original_features = prepare_features(clean_data)
    
    # Step 4: Find optimal clusters
    print("\nDetermining optimal number of clusters...")
    wcss, silhouette_scores = find_optimal_clusters(scaled_features)
    
    # Based on the plots
    optimal_clusters = 4  
    
    # Step 5: Perform clustering
    print(f"\nPerforming clustering with {optimal_clusters} clusters...")
    cluster_labels, kmeans_model = perform_clustering(scaled_features, optimal_clusters)
    
    # Step 6: Analyze and visualize
    print("\nAnalyzing and visualizing results...")
    clustered_data = analyze_and_visualize(clean_data, cluster_labels, scaled_features)
    
    # Save the clustered data
    clustered_data.to_csv('clustered_customers.csv', index=False)
    print("\nClustered data saved to 'clustered_customers.csv'")
    
    print("\n=== Project Completed Successfully ===")