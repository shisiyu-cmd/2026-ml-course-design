import pandas as pd, os
csv_path = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
test_dir = r"C:\Users\古恩宇\Downloads\2026task2\fer_data\fer_data\test"

df = pd.read_csv(csv_path)
print("shape:", df.shape)
print("columns:", list(df.columns))
print("dtypes:", df.dtypes.to_dict())
print("label range:", df['Emotion'].min(), "-", df['Emotion'].max())
print("nulls:", df.isnull().sum().to_dict())
print("duplicated IDs:", df['ID'].duplicated().sum())

# compare with test files
test_files = set(os.listdir(test_dir))
csv_ids = set(df['ID'])
print("ids in csv but not in test:", len(csv_ids - test_files))
print("ids in test but not in csv:", len(test_files - csv_ids))
print("all match:", csv_ids == test_files)

print("\nhead:")
print(df.head())
print("\ntail:")
print(df.tail())
print("\nlabel counts:")
print(df['Emotion'].value_counts().sort_index())
