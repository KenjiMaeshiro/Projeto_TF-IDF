# Análise de Reviews da Steam utilizando TF-IDF

Projeto da disciplina de **Álgebra Linear** — Curso de Ciência de Dados, FATEC Rubens Lara (Santos-SP), 2º semestre de 2025.

**Autor:** Caio Kenji de Paula Maeshiro

## Descrição

Este projeto analisa avaliações (reviews) públicas de usuários da plataforma **Steam**, com o objetivo de identificar semelhanças semânticas entre a primeira avaliação registrada de um jogo e as demais, utilizando a técnica **TF-IDF (Term Frequency – Inverse Document Frequency)** combinada com **similaridade do cosseno**.

O jogo escolhido para o estudo foi **Call of Duty: Modern Warfare 2 (2009)** (Game ID `10180`), filtrado a partir de um dataset com mais de 6,4 milhões de reviews em inglês.

## Dataset

O dataset original contém as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `Review text` | Texto livre da avaliação feita pelo usuário |
| `Game ID` | Identificador numérico do jogo |
| `Sentiment` | Sentimento da review (positivo/negativo) |
| `Helpful` | Número de usuários que marcaram a review como útil |

> O arquivo original está em formato CSV compactado. Para este projeto, os dados foram filtrados apenas para o `Game ID = 10180`.

## Etapas do Projeto

1. **Leitura e limpeza dos dados**
   - Remoção de colunas irrelevantes e renomeação de colunas (`id_jogo`, `Review`).
   - Filtragem pelo `Game ID` 10180.
   - Conversão das reviews para string, remoção de duplicatas e reindexação do DataFrame.

2. **Pré-processamento de texto** (`limpar_review`)
   - Conversão para letras minúsculas.
   - Remoção de símbolos, pontuação, números e códigos HTML.
   - Tokenização (divisão em palavras).
   - Remoção de *stopwords* em inglês (biblioteca `NLTK`).
   - Reagrupamento do texto limpo em uma nova coluna `Review_limpa`.

3. **Vetorização TF-IDF**
   - Uso do `TfidfVectorizer` (scikit-learn) para transformar as reviews limpas em uma matriz esparsa (`reviews_matrix`).

4. **Cálculo de similaridade**
   - Aplicação da `cosine_similarity` entre a primeira review (índice 0) e todas as demais.
   - Ordenação decrescente das similaridades e exibição das 10 reviews mais próximas.

## Tecnologias utilizadas

- Python
- [pandas](https://pandas.pydata.org/)
- [NLTK](https://www.nltk.org/) (stopwords)
- [scikit-learn](https://scikit-learn.org/) (`TfidfVectorizer`, `cosine_similarity`)

## Como executar

```bash
pip install pandas nltk scikit-learn
```

```python
import nltk
nltk.download("stopwords")
```

```bash
python analise_tfidf.py
```

O script imprime no terminal a review base (índice 0) e as 10 reviews mais semelhantes, com o respectivo grau de similaridade.

## Resultados

A review base utilizada como referência foi:

> "makarov thought saw ghost got spooked dropped soap soon paid price"

As 10 reviews mais similares apresentaram valores de similaridade entre **0.17 e 0.24**, um nível moderado — esperado, já que as reviews são curtas, informais e com vocabulário variado.

| Posição | Índice | Similaridade |
|---|---|---|
| 1 | 2961 | 0.24 |
| 2 | 1492 | 0.24 |
| 3 | 2681 | 0.23 |
| 4 | 1759 | 0.20 |
| 5 | 1129 | 0.19 |
| 6 | 3416 | 0.18 |
| 7 | 2581 | 0.18 |
| 8 | 4731 | 0.18 |
| 9 | 3655 | 0.17 |
| 10 | 5065 | 0.17 |

Termos como **ghost**, **soap**, **price** e **makarov** — referências a personagens e eventos do jogo — receberam maior peso no TF-IDF por sua frequência nos textos, sendo os principais responsáveis pelas semelhanças identificadas.

## Conclusão

O projeto demonstrou a eficácia da combinação TF-IDF + similaridade do cosseno para identificar padrões e termos comuns entre reviews curtas e informais de usuários, mesmo em um cenário de vocabulário variado e baixa sobreposição lexical direta.

## Relatório completo

O relatório completo do projeto (PDF) está disponível neste repositório.
