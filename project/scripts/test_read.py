import pandas as pd
print('before read')
df = pd.read_csv('data/raw/amazon_products.csv', nrows=5)
print(df.head())
