import pandas as pd

df = pd.read_csv("data/crawler/collected/fraser_sample.csv")
sampled = df["body"].dropna().sample(n=3, random_state=None)

for i, text in enumerate(sampled, 1):
    print(f"\n[{i}]")
    print(text)