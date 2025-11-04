import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("reviews_filtradas.csv", sep=',', decimal=',')
df = df.drop(columns=["1", "0"])
df = df.rename(columns={"10": "id_jogo", "Ruined my life.": "Review"})
df = df[df["id_jogo"] == 10180]
df["Review"] = df["Review"].astype(str)
df = df.reset_index(drop=True)
print(df.head())

nltk.download("stopwords")
def limpar_review(review):
    review = review.lower()
    review = re.sub(r"&[a-z]+;", " ", review)
    review = re.sub(r"[^a-z\s]", "", review)
    palavras = review.split()
    palavras = [p for p in palavras if p not in stopwords.words("english")]
    return " ".join(palavras)
df["Review_limpa"] = df["Review"].apply(limpar_review)
print(df[["Review", "Review_limpa"]].head())

vetorizador = TfidfVectorizer()
matrix = vetorizador.fit_transform(df["Review_limpa"])
indice = 0
similaridade = cosine_similarity(matrix[indice], matrix)
similaridades = list(enumerate(similaridade[indice]))
similaridades = sorted(similaridades, key=lambda x: x[1], reverse = True)

print(f"\nBase Review 0: {df['Review_limpa'][0]}")
print(f"\nTop 10 reviews mais parecidas:")
for i, score in similaridades[1:11]:
    print(f"\nReview {i}: \n{df["Review_limpa"][i]} \n(similaridade: {score:.2f})")

