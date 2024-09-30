import pandas as pd
import numpy as np

# Load the dataset
file_path = r'D:\online_retail.xlsx'
data = pd.read_excel(file_path)
# Remove rows where Customer ID is NaN
data = data.dropna(subset=['Customer ID'])

# Convert Customer ID to integer
data['Customer ID'] = data['Customer ID'].astype(int)

# Remove cancelled transactions
data = data[~data['Invoice'].str.contains('C', na=False)]
# Calculate Total Price
data['TotalPrice'] = data['Quantity'] * data['Price']

# Create date features
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['YearMonth'] = data['InvoiceDate'].dt.to_period('M')

# Aggregate data by Customer ID
customer_data = data.groupby('Customer ID').agg({
    'TotalPrice': 'sum',        # Total spend
    'Invoice': 'nunique',       # Total number of unique visits (orders)
    'Quantity': 'sum',          # Total items purchased
    'InvoiceDate': 'max'        # Last purchase date
}).rename(columns={
    'Invoice': 'TotalOrders',
    'Quantity': 'TotalItems',
    'InvoiceDate': 'LastPurchaseDate'
})

# Calculate Days since last purchase
latest_date = data['InvoiceDate'].max()
customer_data['DaysSinceLastPurchase'] = (latest_date - customer_data['LastPurchaseDate']).dt.days

# Drop the LastPurchaseDate as it's no longer needed
customer_data.drop('LastPurchaseDate', axis=1, inplace=True)
# Display the first few rows of the processed data
print(customer_data.head())
# Save the customer_data DataFrame to a CSV file
# Specify the path on the D drive where you want to save the file
file_path = 'D:/customer_data.csv'

# Save the customer_data DataFrame to a CSV file at the specified location
customer_data.to_csv(file_path, index_label='Customer ID')

