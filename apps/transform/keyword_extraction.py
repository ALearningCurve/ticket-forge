from sklearn.feature_extraction.text import TfidfVectorizer


_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
)


def extract_keywords(texts, top_k=10):
    tfidf = _vectorizer.fit_transform(texts)
    feature_names = _vectorizer.get_feature_names_out()

    keywords = []
    for row in tfidf:
        indices = row.toarray()[0].argsort()[-top_k:][::-1]
        keywords.append([feature_names[i] for i in indices])

    return keywords
