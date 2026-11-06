import pandas as pd
csv_path = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
tpl_path = r"C:\Users\古恩宇\Downloads\submission.csv"

df = pd.read_csv(csv_path)
tpl = pd.read_csv(tpl_path)

print("shape:", df.shape)
print("columns:", list(df.columns))
print("label range:", df['Emotion'].min(), "-", df['Emotion'].max())
print("nulls:", df.isnull().sum().to_dict())
print("duplicated IDs:", df['ID'].duplicated().sum())

# order match
order_match = (list(df['ID']) == list(tpl['ID']))
print("ID order matches template:", order_match)

print("\nhead:")
print(df.head())
print("\nlabel counts:")
print(df['Emotion'].value_counts().sort_index())
