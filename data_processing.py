import pandas as pd

df1 = pd.read_csv('2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')
df2 = pd.read_csv('2020-12-31-DynamicallyGeneratedHateDataset-targets-v0.1.csv')

df1['id'] = df1['id'].astype(str).str.strip()
df2['id'] = df2['id'].astype(str).str.strip()

target_columns = df2.columns.drop(['id'])
target_counts = df2[target_columns].sum(axis=1)
is_multiple_targets = target_counts > 1
is_other_flagged = df2['other'] == 1
df2['is_intersectional'] = (is_multiple_targets | is_other_flagged).astype(int)

intersectional_count = (df2['is_intersectional'] == 1).sum()
print(f"Number of intersectional rows: {intersectional_count}")

merged_df = pd.merge(df1, df2, on='id', how='inner')
merged_df.to_csv('processed_data.csv', index=False)

# 724 / 40623 = 1.78% of samples are labeled as intersectional hate
print("Merged CSV created successfully with shape:", merged_df.shape)