import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the processed data
customer_data = pd.read_csv('D:/processed_customer_data.csv', index_col='Customer ID')

# Load the scaler object
scaler = load('D:/customer_data_scaler.joblib')

# Apply the scaler to the features
features = customer_data[['TotalPrice', 'TotalOrders', 'DaysSinceLastPurchase']]
scaled_features = scaler.transform(features)

# K-means Clustering
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Assuming the elbow is observed at k=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

# Analyze and profile clusters
cluster_summary = customer_data.groupby('Cluster').agg({
    'TotalPrice': ['mean', 'min', 'max'],
    'TotalOrders': ['mean', 'min', 'max'],
    'DaysSinceLastPurchase': ['mean', 'min', 'max']
})
print("Cluster Summary:\n", cluster_summary)

# Visualization of clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=customer_data['Cluster'], cmap='viridis', label=customer_data['Cluster'])
plt.title('Customer Segments Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
# Save the DataFrame with cluster labels to a CSV file
customer_data.to_csv('D:/customer_data_with_clusters.csv', index=True, index_label='Customer ID')
# Assuming 'kmeans' is your KMeans clustering model
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['TotalPrice', 'TotalOrders', 'DaysSinceLastPurchase'])
centroids.to_csv('D:/cluster_centroids.csv', index=True, index_label='Cluster')
