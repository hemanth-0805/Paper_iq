import io
import os
import re
import math
from datetime import datetime  # noqa: F401 (kept for parity with original)
from typing import Dict, List, Tuple, Optional
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from PyPDF2 import PdfReader

# Optional: better PDF text extraction than PyPDF2
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except Exception:
    fitz = None
    PYMUPDF_AVAILABLE = False

# Optional: transformer summarizer (best-effort; app must still work without it)
try:
    from transformers import pipeline as hf_pipeline

    TRANSFORMERS_AVAILABLE = True
except Exception:
    hf_pipeline = None
    TRANSFORMERS_AVAILABLE = False


def _clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(uploaded_file) -> Tuple[str, str]:
    """Return (paper_name, extracted_text)."""
    if uploaded_file is None:
        return "", ""

    paper_name = uploaded_file.name
    ext = os.path.splitext(paper_name)[1].lower()

    if ext == ".txt":
        raw = uploaded_file.read()
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            text = str(raw)
        return paper_name, _clean_text(text)

    if ext == ".pdf":
        file_bytes = uploaded_file.read()
        # Prefer PyMuPDF when available for higher-fidelity extraction.
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text_blocks = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    # Keep text blocks; sort top-to-bottom then left-to-right.
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    text_blocks.extend([b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) > 4])
                text = "\n".join(text_blocks)
                # De-hyphenate line breaks like "algo-\nrithm"
                text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
                return paper_name, _clean_text(text)
            except Exception:
                # Fallback to PyPDF2 if PyMuPDF fails on a file.
                pass

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        return paper_name, _clean_text(text)

    return paper_name, ""


@st.cache_resource(show_spinner=False)
def load_summarizer():
    if not TRANSFORMERS_AVAILABLE or hf_pipeline is None:
        return None
    try:
        # Use a small model; still may fail on some environments (we fallback gracefully).
        return hf_pipeline("summarization", model="t5-small")
    except Exception:
        return None


SUMMARIZER = load_summarizer()


class InsightEngine:
    """
    Reference-style engine:
    - Extracts/cleans text
    - Detects headings and assigns content to sections
    - Infers missing core sections
    - Generates 3-level summaries per section (AI if available; else heuristic)
    - Computes scores (Language/Coherence/Reasoning/Sophistication/Readability/Composite)
    """

    def __init__(self):
        self.full_text: str = ""
        self.clean_text_content: str = ""

        # Raw detected headings (as they appear) -> extracted text
        self.sections_detected: Dict[str, str] = {}
        # AI/heuristic summaries for each validated detected heading
        # title -> {"Short","Medium","Long"}
        self.detected_section_summaries: Dict[str, Dict[str, str]] = {}

        # Canonical core sections
        self.mandatory_map = {
            "Abstract": "",
            "Introduction": "",
            "Methodology": "",
            "Results": "",
            "Conclusion": "",
        }
        self.section_detected_flag = {k: False for k in self.mandatory_map}
        self.ai_summaries = {k: {"Short": "N/A", "Medium": "N/A", "Long": "N/A"} for k in self.mandatory_map}

        self.stats: Dict[str, float] = {}
        self.word_freq: List[Tuple[str, int]] = []
        self.domain: str = "General Research"

        self.scores: Dict[str, float] = {}
        self.sentiment: float = 0.0
        self.issues: List[str] = []

        self.paper_summaries = {"Short": "", "Medium": "", "Long": ""}
        self.paper_tldr: str = ""

    def clean_text_func(self, text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def analyze_frequency(self, text: str) -> List[Tuple[str, int]]:
        words = (text or "").split()
        stop_words = {
            "the",
            "and",
            "of",
            "to",
            "in",
            "a",
            "is",
            "that",
            "for",
            "on",
            "with",
            "as",
            "by",
            "at",
            "this",
            "from",
            "it",
            "be",
            "are",
            "which",
            "an",
            "or",
        }
        filtered = [w for w in words if w not in stop_words and not w.isdigit() and len(w) > 2]
        return Counter(filtered).most_common(20)

    def classify_domain(self, text: str) -> str:
        t = (text or "").lower()
        if any(x in t for x in ["neural network", "deep learning", "ai", "machine learning"]):
            return "Artificial Intelligence"
        if any(x in t for x in ["iot", "sensor", "wireless", "embedded"]):
            return "Internet of Things"
        if any(x in t for x in ["data mining", "sentiment", "classification", "regression"]):
            return "Data Science"
        return "General Research"

    def _smart_infer(self, text: str, keywords: List[str]) -> str:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        candidates: List[Tuple[int, str]] = []
        for p in paragraphs:
            score = sum(1 for k in keywords if k in p.lower())
            if score > 0:
                candidates.append((score, p))
        candidates.sort(key=lambda x: x[0], reverse=True)
        if candidates:
            return "\n\n".join([c[1] for c in candidates[:3]]).strip()
        return "Content inferred based on document context."

    def _heuristic_3_summaries(self, text: str) -> Dict[str, str]:
        sents = [s.strip() for s in sent_tokenize(text or "") if s.strip()]
        if not sents:
            t = (text or "").strip()
            return {"Short": t, "Medium": t, "Long": t}
        return {
            "Short": (" ".join(sents[:2]) + ("..." if len(sents) > 2 else "")).strip(),
            "Medium": (" ".join(sents[:4]) + ("..." if len(sents) > 4 else "")).strip(),
            "Long": (" ".join(sents[:8]) + ("..." if len(sents) > 8 else "")).strip(),
        }

    def _generate_3_summaries(self, text: str) -> Dict[str, str]:
        text = (text or "").strip()
        if not text or len(text) < 80:
            return {"Short": text, "Medium": text, "Long": text}

        if SUMMARIZER is not None:
            try:
                input_text = text[:1500]
                s = SUMMARIZER(input_text, max_length=50, min_length=15, do_sample=False)
                m = SUMMARIZER(input_text, max_length=120, min_length=50, do_sample=False)
                l = SUMMARIZER(input_text, max_length=250, min_length=100, do_sample=False)
                return {
                    "Short": (s[0].get("summary_text") or "").strip(),
                    "Medium": (m[0].get("summary_text") or "").strip(),
                    "Long": (l[0].get("summary_text") or "").strip(),
                }
            except Exception:
                return self._heuristic_3_summaries(text)

        return self._heuristic_3_summaries(text)

    def _syllable_count(self, word: str) -> int:
        word = (word or "").lower()
        count = 0
        vowels = "aeiouy"
        prev_is_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1
        return max(1, count)

    def _coherence_score(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 100.0
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
            tfidf = vectorizer.fit_transform([s.lower() for s in sentences])
            sims = []
            for i in range(len(sentences) - 1):
                sim = (tfidf[i] * tfidf[i + 1].T).toarray()[0][0]
                sims.append(float(sim))
            avg_sim = float(sum(sims) / len(sims)) if sims else 0.0
            return float(max(0.0, min(100.0, avg_sim * 100.0)))
        except Exception:
            return 50.0

    def _reasoning_score(self, sentences: List[str]) -> float:
        indicators = [
            "because",
            "since",
            "if",
            "then",
            "implies",
            "leads to",
            "causes",
            "therefore",
            "thus",
            "hence",
            "consequently",
            "as a result",
            "in order to",
            "so that",
            "due to",
            "for this reason",
        ]
        indicator_sentences = 0
        for s in sentences:
            s_l = s.lower()
            if any(ind in s_l for ind in indicators):
                indicator_sentences += 1

        pattern_score = 0.0
        patterns = [
            r"\b\w+\s+causes?\s+\w+",
            r"\b\w+\s+leads?\s+to\s+\w+",
            r"\bif\s+.+\s+then\s+.+",
            r"\bdue\s+to\s+.+",
        ]
        for p in patterns:
            if re.search(p, (self.full_text or "").lower()):
                pattern_score += 20.0

        ratio = (indicator_sentences / len(sentences)) if sentences else 0.0
        raw = ratio * 60.0 + min(pattern_score, 40.0)
        return float(min(85.0, raw))

    def _language_score(self, words: List[str], sentence_count: int) -> float:
        word_count = len(words)
        if word_count == 0 or sentence_count == 0:
            return 0.0
        unique_words = set(w.lower() for w in words)
        vocab_div = len(unique_words) / word_count
        vocab_score = vocab_div * 50.0

        avg_sent_len = word_count / sentence_count
        if avg_sent_len < 15:
            sent_len_score = (avg_sent_len / 15.0) * 25.0
        elif avg_sent_len > 25:
            sent_len_score = max(0.0, 25.0 - (avg_sent_len - 25.0) * 2.0)
        else:
            sent_len_score = 25.0

        avg_word_len = sum(len(w) for w in words) / word_count
        word_len_score = min(avg_word_len / 10.0, 1.0) * 15.0

        complexity = (self.full_text or "").count(",") + (self.full_text or "").count(";") + (self.full_text or "").count(":")
        complexity_score = min((complexity / sentence_count) * 10.0, 10.0)

        return float(vocab_score + sent_len_score + word_len_score + complexity_score)

    def _sophistication_score(self, words: List[str]) -> float:
        word_count = len(words)
        if word_count == 0:
            return 0.0
        avg_word_len = sum(len(w) for w in words) / word_count
        word_len_component = min(avg_word_len / 10.0, 1.0) * 50.0

        unique_words = set(w.lower() for w in words)
        vocab_div = len(unique_words) / word_count
        diversity_component = vocab_div * 50.0

        return float(word_len_component * 0.6 + diversity_component * 0.4)

    def _readability_score(self, words: List[str], sentence_count: int) -> float:
        word_count = len(words)
        if word_count == 0 or sentence_count == 0:
            return 50.0
        total_syllables = sum(self._syllable_count(w) for w in words)
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (total_syllables / word_count)
        return float(max(0.0, min(100.0, score)))

    def compute_scores(self):
        sentences = [s.strip() for s in sent_tokenize(self.full_text or "") if s.strip()]
        words = re.findall(r"\b[a-zA-Z]+\b", self.full_text or "")
        word_count = len(words)
        sentence_count = len(sentences)

        if word_count == 0 or sentence_count == 0:
            self.scores = {k: 0.0 for k in ["Language", "Coherence", "Reasoning", "Sophistication", "Readability", "Composite"]}
            self.sentiment = 0.0
            self.issues = []
            return

        avg_sentence_len = word_count / sentence_count
        avg_word_len = sum(len(w) for w in words) / word_count
        unique_words = set(w.lower() for w in words)
        vocab_div = len(unique_words) / word_count
        complex_words = [w for w in words if len(w) > 6]
        complex_ratio = len(complex_words) / word_count

        self.stats.update(
            {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_len": avg_sentence_len,
                "avg_word_len": avg_word_len,
                "vocab_diversity": vocab_div,
                "complex_word_ratio": complex_ratio,
            }
        )

        coherence = self._coherence_score(sentences)
        reasoning = self._reasoning_score(sentences)
        language = min(100.0, max(0.0, self._language_score(words, sentence_count)))
        sophistication = self._sophistication_score(words)
        readability = self._readability_score(words, sentence_count)

        composite = language * 0.2 + coherence * 0.25 + reasoning * 0.2 + sophistication * 0.15 + readability * 0.2
        self.scores = {
            "Language": float(language),
            "Coherence": float(coherence),
            "Reasoning": float(reasoning),
            "Sophistication": float(sophistication),
            "Readability": float(readability),
            "Composite": float(composite),
        }

        # Issues: long sentences
        self.issues = [s for s in sentences if len(s.split()) > 30]

        # Sentiment: optional (TextBlob) if installed; otherwise neutral.
        try:
            from textblob import TextBlob  # type: ignore

            self.sentiment = float(TextBlob(self.full_text).sentiment.polarity)
        except Exception:
            self.sentiment = 0.0

    def _generate_suggestions(self) -> List[str]:
        weak_phrases = {
            r"\bshows?\b": "demonstrates",
            r"\bproves?\b": "confirms",
            r"\bgood\b": "beneficial",
            r"\bbad\b": "adverse",
            r"\bbig\b": "substantial",
            r"\bvery\b": "extremely",
            r"\bthis paper\b": "this study",
            r"\bin this paper\b": "in this work",
            r"\bwe do\b": "we perform",
            r"\bwe see\b": "we observe",
        }
        suggestions = []
        lower = (self.full_text or "").lower()
        for pat, repl in weak_phrases.items():
            if re.search(pat, lower):
                suggestions.append(f"Consider using '{repl}' instead of phrases matching '{pat}'.")
        return suggestions[:5]

    def process_text(self, raw_text: str):
        self.full_text = raw_text or ""
        self._analyze(page_count=1)

    def process_pdf_bytes(self, file_bytes: bytes):
        # Extract again here (InsightEngine reference); but we already extracted in `extract_text`.
        # Keep this method for future multi-file support.
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                text_blocks = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    blocks.sort(key=lambda b: (b[1], b[0]))
                    text_blocks.extend([b[4] for b in blocks if isinstance(b, (list, tuple)) and len(b) > 4])
                text = "\n".join(text_blocks)
                text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
                self.full_text = text
                self._analyze(page_count=len(doc))
                return
            except Exception:
                pass

        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        self.full_text = "\n".join(pages)
        self._analyze(page_count=len(reader.pages))

    def _analyze(self, page_count: int):
        self.full_text = _clean_text(self.full_text)
        self.clean_text_content = self.clean_text_func(self.full_text)

        header_regex = r"(?m)^(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][a-zA-Z0-9\s\-\:]+)\s*$"
        lines = (self.full_text or "").split("\n")

        current_header = None
        current_buffer: List[str] = []

        def map_header(h: str) -> Optional[str]:
            hh = (h or "").lower().strip()
            if "abstract" in hh or "summary" in hh:
                return "Abstract"
            if "introduction" in hh or "background" in hh:
                return "Introduction"
            if any(x in hh for x in ["method", "proposed", "implementation", "approach", "materials"]):
                return "Methodology"
            if any(x in hh for x in ["result", "experiment", "evaluation", "findings"]):
                return "Results"
            if "conclusion" in hh or "future work" in hh:
                return "Conclusion"
            return None

        def commit_section(header: str, buf: List[str]):
            text_content = "\n".join(buf).strip()
            if len(text_content) <= 50:
                return
            self.sections_detected[header] = text_content
            mapped_key = map_header(header)
            if mapped_key:
                self.mandatory_map[mapped_key] = (self.mandatory_map[mapped_key] + "\n\n" + text_content).strip()
                self.section_detected_flag[mapped_key] = True

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            is_headerish = len(line) < 80 and re.match(header_regex, line)
            if is_headerish:
                if current_header:
                    commit_section(current_header, current_buffer)
                current_header = re.sub(r"^[\d\.\s]+", "", line).strip()
                current_buffer = []
            else:
                if current_header:
                    current_buffer.append(line)

        if current_header and current_buffer:
            commit_section(current_header, current_buffer)

        # Infer missing core sections
        if not self.mandatory_map["Abstract"]:
            self.mandatory_map["Abstract"] = (self.full_text or "")[:1000]
        if not self.mandatory_map["Introduction"]:
            self.mandatory_map["Introduction"] = self._smart_infer(self.full_text, ["introduction", "background", "overview"])
        if not self.mandatory_map["Methodology"]:
            self.mandatory_map["Methodology"] = self._smart_infer(self.full_text, ["method", "proposed", "algorithm", "system"])
        if not self.mandatory_map["Results"]:
            self.mandatory_map["Results"] = self._smart_infer(self.full_text, ["result", "performance", "accuracy", "table"])
        if not self.mandatory_map["Conclusion"]:
            self.mandatory_map["Conclusion"] = self._smart_infer(self.full_text, ["conclusion", "summary", "future"])

        # Summaries per core section
        for key in self.mandatory_map:
            content = self.mandatory_map[key]
            self.ai_summaries[key] = self._generate_3_summaries(content) if content and len(content) > 50 else {"Short": "N/A", "Medium": "N/A", "Long": "N/A"}

        # Validate each detected heading and generate summaries just for valid ones.
        self.detected_section_summaries = {}
        for title, body in self.sections_detected.items():
            word_count = len(re.findall(r"\b\w+\b", body))
            if word_count < 50:
                continue
            if not re.search(r"[A-Za-z]{2,}", title):
                continue
            self.detected_section_summaries[title] = self._generate_3_summaries(body)

        combined_source = (self.mandatory_map["Abstract"] + "\n" + self.mandatory_map["Introduction"]).strip()
        self.paper_summaries = self._generate_3_summaries(combined_source)
        self.paper_tldr = self.paper_summaries.get("Medium", "")

        self.word_freq = self.analyze_frequency(self.clean_text_content)
        self.domain = self.classify_domain(self.clean_text_content)

        words_list = self.clean_text_content.split()
        self.stats.update(
            {
                "pages": float(page_count),
                "sections": float(len(self.sections_detected)),
                "words": float(len(words_list)),
                "unique_words": float(len(set(words_list))),
                "time": f"{math.ceil(len(words_list) / 200)} min" if words_list else "0 min",
            }
        )

        self.compute_scores()


def extract_keywords(text: str, top_k: int = 20) -> pd.DataFrame:
    text = text or ""
    sw = set(stopwords.words("english")) if nltk else set()
    vectorizer = TfidfVectorizer(
        stop_words=list(sw) if sw else "english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    try:
        tfidf = vectorizer.fit_transform([text])
    except ValueError:
        return pd.DataFrame(columns=["keyword", "score"])

    scores = tfidf.toarray()[0]
    feats = np.array(vectorizer.get_feature_names_out())
    idx = np.argsort(scores)[::-1][:top_k]
    data = [{"keyword": feats[i], "score": float(scores[i])} for i in idx if scores[i] > 0]
    return pd.DataFrame(data)


def compare_papers(text_a: str, text_b: str) -> Dict:
    """
    Compare two papers using TF-IDF cosine similarity + keyword overlap + overall summaries.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
    X = vectorizer.fit_transform([text_a or "", text_b or ""])
    denom = float(np.linalg.norm(X[0].toarray()) * np.linalg.norm(X[1].toarray())) or 1e-12
    sim = float((X[0] * X[1].T).toarray()[0][0] / denom)

    kw_a = set(extract_keywords(text_a, top_k=30)["keyword"].tolist())
    kw_b = set(extract_keywords(text_b, top_k=30)["keyword"].tolist())
    overlap = sorted(list(kw_a.intersection(kw_b)))

    eng_a = InsightEngine()
    eng_a.process_text(text_a or "")
    eng_b = InsightEngine()
    eng_b.process_text(text_b or "")
    sum_a = eng_a.paper_summaries.get("Medium", "") or eng_a.paper_tldr
    sum_b = eng_b.paper_summaries.get("Medium", "") or eng_b.paper_tldr

    # Compare metrics across both papers
    compare_metrics = ["Composite", "Readability", "Coherence", "Reasoning", "Language", "Sophistication"]
    best_by = {}
    a_count = 0
    b_count = 0
    for metric in compare_metrics:
        a_score = float(eng_a.scores.get(metric, 0.0))
        b_score = float(eng_b.scores.get(metric, 0.0))
        if abs(a_score - b_score) < 1e-4:
            best_by[metric] = "Tie"
        elif a_score > b_score:
            best_by[metric] = "Paper A"
            a_count += 1
        else:
            best_by[metric] = "Paper B"
            b_count += 1

    if a_count > b_count:
        best_overall = "Paper A"
    elif b_count > a_count:
        best_overall = "Paper B"
    else:
        best_overall = "Tie"

    return {
        "similarity": sim,
        "keyword_overlap": overlap[:25],
        "summary_a": sum_a,
        "summary_b": sum_b,
        "scores_a": eng_a.scores,
        "scores_b": eng_b.scores,
        "best_by": best_by,
        "best_overall": best_overall,
    }

