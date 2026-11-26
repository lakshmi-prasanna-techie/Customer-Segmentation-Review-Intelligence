import re
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from config import (
    BASE_NUMERIC_COLS,
    COMPLAINT_KEYWORDS,
    NEGATIVE_WORDS,
    POSITIVE_WORDS,
    RANDOM_STATE,
    SVD_COMPONENTS,
    TFIDF_MAX_FEATURES,
)


def compute_lexicon_features(text_series: pd.Series) -> pd.DataFrame:
    sentiments, complaint_rates, word_counts = [], [], []
    for raw in text_series.fillna(""):
        tokens = re.findall(r"[a-z]+", str(raw).lower())
        total = max(len(tokens), 1)
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        complaints = sum(1 for t in tokens if t in COMPLAINT_KEYWORDS)

        sentiments.append((pos - neg) / max(pos + neg, 1))
        complaint_rates.append(complaints / total)
        word_counts.append(len(tokens))

    return pd.DataFrame({
        "nlp_sentiment_score": sentiments,
        "nlp_complaint_rate": complaint_rates,
        "review_word_count": word_counts,
    })


def build_feature_matrix(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer | None = None,
    svd: TruncatedSVD | None = None,
    fit: bool = True,
):
    work_df = df.copy().reset_index(drop=True)
    for col in BASE_NUMERIC_COLS:
        if col not in work_df.columns:
            work_df[col] = 0.0
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    if "review_text" not in work_df.columns:
        work_df["review_text"] = ""

    work_df["review_count"] = work_df["review_count"].fillna(0.0)
    work_df["order_frequency"] = work_df["order_frequency"].fillna(0.0)
    for col in ["recency_days", "avg_order_value", "avg_star_rating"]:
        work_df[col] = work_df[col].fillna(work_df[col].median())

    work_df["estimated_clv_proxy"] = work_df["order_frequency"] * work_df["avg_order_value"]

    lex_df = compute_lexicon_features(work_df["review_text"])
    work_df = pd.concat([work_df, lex_df], axis=1)

    clean_text = work_df["review_text"].fillna("").astype(str)
    if fit:
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_mat = tfidf.fit_transform(clean_text)
        n_comp = min(SVD_COMPONENTS, max(1, tfidf_mat.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
        svd_feats = svd.fit_transform(tfidf_mat)
    else:
        tfidf_mat = tfidf.transform(clean_text)
        svd_feats = svd.transform(tfidf_mat)

    svd_cols = [f"review_topic_{i + 1}" for i in range(svd_feats.shape[1])]
    svd_df = pd.DataFrame(svd_feats, columns=svd_cols)
    work_df = pd.concat([work_df, svd_df], axis=1)

    feature_cols = (
        BASE_NUMERIC_COLS
        + ["estimated_clv_proxy", "nlp_sentiment_score", "nlp_complaint_rate", "review_word_count"]
        + svd_cols
    )
    return work_df, feature_cols, tfidf, svd
