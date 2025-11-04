import pandas as pd

df = pd.read_csv("steam.csv", sep=',', decimal=',')
df = df.head(1000000)
df.to_csv("reviews_filtradas.csv", index=False, sep=',', decimal=',')