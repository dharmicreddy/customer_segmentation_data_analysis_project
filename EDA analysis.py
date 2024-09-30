import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load the customer data from the CSV file on D Drive
customer_data = pd.read_csv('D:/customer_data.csv', index_col='Customer ID')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
# Get a statistical summary of the numerical columns
print(customer_data.describe())
# Plot the distribution of total spend
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['TotalPrice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Total Spend')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of total orders
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['TotalOrders'], kde=False, bins=30, color='green')
plt.title('Distribution of Total Orders')
plt.xlabel('Total Orders')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of days since last purchase
plt.figure(figsize=(10, 6))
sns.histplot(customer_data['DaysSinceLastPurchase'], kde=False, bins=30, color='red')
plt.title('Days Since Last Purchase')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()
# Box plot for total spend
plt.figure(figsize=(10, 6))
sns.boxplot(x=customer_data['TotalPrice'])
plt.title('Box Plot of Total Spend')
plt.show()

# Box plot for total orders
plt.figure(figsize=(10, 6))
sns.boxplot(x=customer_data['TotalOrders'])
plt.title('Box Plot of Total Orders')
plt.show()
# Compute the correlation matrix
corr = customer_data.corr()

# Generate a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Matrix of Customer Data')
plt.show()

# Assuming customer_data includes all necessary transformations
customer_data.to_csv('D:/processed_customer_data.csv', index=True, index_label='Customer ID')

# Assume we're scaling TotalPrice, TotalOrders, and DaysSinceLastPurchase
scaler = StandardScaler()
customer_features = customer_data[['TotalPrice', 'TotalOrders', 'DaysSinceLastPurchase']]
scaler.fit(customer_features)

# Save the scaler
dump(scaler, 'D:/customer_data_scaler.joblib')
