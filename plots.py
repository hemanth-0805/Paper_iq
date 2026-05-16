from typing import Dict
import re

import pandas as pd
import numpy as np  # noqa: F401 (kept for compatibility with existing environment)
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def plot_word_freq(text: str, top_n: int = 20):
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{2,}\b", text or "")]
    sw = set(stopwords.words("english"))
    words = [w for w in words if w not in sw]
    if not words:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No words to plot", ha="center", va="center")
        ax.axis("off")
        return fig

    ser = pd.Series(words).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=ser.values, y=ser.index, ax=ax, palette="viridis")
    ax.set_title("Top Word Frequency")
    ax.set_xlabel("Count")
    ax.set_ylabel("Word")
    fig.tight_layout()
    return fig


def plot_sentence_length_distribution(text: str):
    sents = [s.strip() for s in sent_tokenize(text or "") if s.strip()]
    lens = [len(re.findall(r"\b\w+\b", s)) for s in sents]
    fig, ax = plt.subplots(figsize=(10, 4))
    if not lens:
        ax.text(0.5, 0.5, "No sentences to plot", ha="center", va="center")
        ax.axis("off")
        return fig
    sns.histplot(lens, bins=30, ax=ax, kde=True, color="#4c72b0")
    ax.set_title("Sentence Length Distribution (words)")
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_keyword_importance(kw_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    if kw_df is None or kw_df.empty:
        ax.text(0.5, 0.5, "No keywords extracted", ha="center", va="center")
        ax.axis("off")
        return fig
    df = kw_df.head(15).sort_values("score", ascending=True)
    sns.barplot(x="score", y="keyword", data=df, ax=ax, palette="magma")
    ax.set_title("Keyword Importance (TF-IDF)")
    ax.set_xlabel("TF-IDF score")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_readability_metrics(metrics: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["readability_score", "coherence_score", "sophistication_score"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]
    labels = ["Readability", "Coherence", "Sophistication"]
    sns.barplot(x=labels, y=vals, ax=ax, palette="deep")
    ax.set_ylim(0, max(100.0, max(vals) * 1.15))
    ax.set_title("Readability & Language Metrics")
    ax.set_ylabel("Score")
    fig.tight_layout()
    return fig

