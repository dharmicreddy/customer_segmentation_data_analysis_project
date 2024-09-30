import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the processed data with clusters
customer_data = pd.read_csv('D:/customer_data_with_clusters.csv', index_col='Customer ID')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Analyze and profile clusters
# Summary statistics for each cluster
cluster_summary = customer_data.groupby('Cluster').agg({
    'TotalPrice': ['mean', 'min', 'max'],
    'TotalOrders': ['mean', 'min', 'max'],
    'DaysSinceLastPurchase': ['mean', 'min', 'max']
})
print("Cluster Summary:\n", cluster_summary)

# Save cluster summary to CSV
cluster_summary.to_csv('D:/cluster_summary.csv')

# Adding more in-depth analysis by looking at other features or behaviors
# Detailed profiles of each cluster
detailed_profiles = ""
for i in range(max(customer_data['Cluster']) + 1):
    profile = f"\nCluster {i} Profile:"
    cluster_data = customer_data[customer_data['Cluster'] == i]
    profile += f"\nAverage Total Spend: ${cluster_data['TotalPrice'].mean():,.2f}"
    profile += f"\nAverage Total Orders: {cluster_data['TotalOrders'].mean():.2f}"
    profile += f"\nAverage Days Since Last Purchase: {cluster_data['DaysSinceLastPurchase'].mean():.2f}"
    detailed_profiles += profile

# Save detailed profiles to a text file
with open('D:/detailed_cluster_profiles.txt', 'w') as file:
    file.write(detailed_profiles)

# Visualization of clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
principal_components = pca.fit_transform(customer_data[['TotalPrice', 'TotalOrders', 'DaysSinceLastPurchase']])
plt.figure(figsize=(12, 8))
colors = ['red', 'green', 'blue']  # Modify as needed based on the number of clusters

for i, color in enumerate(colors):
    plt.scatter(principal_components[customer_data['Cluster'] == i, 0],
                principal_components[customer_data['Cluster'] == i, 1],
                color=color,
                label=f'Cluster {i}')
plt.title('Customer Segments Visualized with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Save the PCA plot
plt.savefig('D:/customer_segments_pca.png')
plt.show()
