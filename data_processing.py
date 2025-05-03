import pandas as pd
from sentence_transformers import SentenceTransformer

df1 = pd.read_csv('2020-12-31-DynamicallyGeneratedHateDataset-entries-v0.1.csv')
df2 = pd.read_csv('2020-12-31-DynamicallyGeneratedHateDataset-targets-v0.1.csv')

df1['id'] = df1['id'].astype(str).str.strip()
df2['id'] = df2['id'].astype(str).str.strip()

target_columns = df2.columns.drop(['id'])
target_counts = df2[target_columns].sum(axis=1)
is_multiple_targets = target_counts > 1
is_other_flagged = df2['other'] == 1
is_gaywom = df2['gaywom'] == 1
is_blawom = df2['blawom'] == 1
is_asiwom = df2['asiwom'] == 1
is_muswom = df2['muswom'] == 1
df2['is_intersectional'] = (is_multiple_targets | is_other_flagged | is_gaywom | is_blawom | is_asiwom | is_muswom).astype(int)

intersectional_count = (df2['is_intersectional'] == 1).sum()
print(f"Number of intersectional rows: {intersectional_count}")

merged_df = pd.merge(df1, df2, on='id', how='inner')
merged_df = merged_df.dropna(subset=['text', 'is_intersectional'])
merged_df.to_csv('processed_data.csv', index=False)

texts = merged_df['text'].astype(str)
labels = merged_df['is_intersectional'].astype(int)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts.tolist(), show_progress_bar=True)
embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(embeddings.shape[1])])
embeddings_df['is_intersectional'] = labels.reset_index(drop=True)
embeddings_df.to_csv('sentence_embeddings_with_labels.csv', index=False)

# 1679 / 40623 = 4.1% of samples are labeled as intersectional hate
print("Merged CSV created successfully with shape:", merged_df.shape)
print("Saved dense embeddings with labels to 'sentence_embeddings_with_labels.csv'")