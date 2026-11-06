import pandas as pd

tpl_path = r"C:\Users\古恩宇\Downloads\submission.csv"        # Kaggle 模板
my_path  = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"
out_path = r"C:\Users\古恩宇\Downloads\2026task2\submission.csv"

tpl = pd.read_csv(tpl_path)
mine = pd.read_csv(my_path)

print("template rows:", len(tpl), "cols:", list(tpl.columns))
print("my rows:", len(mine), "cols:", list(mine.columns))

# build id -> emotion map from my predictions
pred_map = dict(zip(mine["ID"], mine["Emotion"]))

# verify every template id is covered
missing = [i for i in tpl["ID"] if i not in pred_map]
extra   = [i for i in pred_map if i not in set(tpl["ID"])]
print("missing in my preds:", len(missing))
print("extra in my preds:", len(extra))

# reorder according to template
out = tpl.copy()
out["Emotion"] = out["ID"].map(pred_map).astype(int)

# sanity
assert out["Emotion"].isnull().sum() == 0, "there are NaN after mapping"
assert list(out["ID"]) == list(tpl["ID"]), "ID order changed!"

out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
print("rows:", len(out))
print("head:")
print(out.head())
print("\nlabel counts:")
print(out["Emotion"].value_counts().sort_index())
