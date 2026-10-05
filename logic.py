"""
logic.py – OmniMat Core AI Engine
Attribute-Aware Conflict Penalty matching algorithm for CPSE material harmonization.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Data & Model Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading material catalogue…")
def load_data() -> pd.DataFrame:
    """Load the material catalogue CSV and normalise missing values."""
    csv_path = os.path.join(os.path.dirname(__file__), "data.csv")
    df = pd.read_csv(csv_path)
    df.fillna("", inplace=True)
    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())
    return df


@st.cache_resource(show_spinner="Loading AI model (one-time)…")
def load_model():
    """Load the sentence-transformer model and cache across sessions."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Embedding Generation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating semantic embeddings…")
def get_embeddings(_df: pd.DataFrame, _model) -> np.ndarray:
    """Encode every raw_description into a dense vector."""
    descriptions = _df["raw_description"].tolist()
    embeddings = _model.encode(descriptions, show_progress_bar=False)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Attribute-Aware Conflict Penalty Matching
# ---------------------------------------------------------------------------

def _attr_match_score(target_row: pd.Series, candidate_row: pd.Series):
    """
    Compare grade and size attributes.

    Returns
    -------
    attr_score : float   0.0 – 1.0   (proportion of matching attributes)
    penalty    : float   0.0 or 1.0   (hard veto if a critical attribute conflicts)
    conflict_details : list[str]      human-readable conflict reasons
    """
    attrs = ["grade", "size"]
    matches = 0
    comparisons = 0
    penalty = 0.0
    conflicts: list[str] = []

    for attr in attrs:
        t_val = str(target_row.get(attr, "")).strip().upper()
        c_val = str(candidate_row.get(attr, "")).strip().upper()

        # Both values present → we can compare
        if t_val and c_val:
            comparisons += 1
            if t_val == c_val:
                matches += 1
            else:
                # Hard veto – critical attribute mismatch
                penalty = 1.0
                conflicts.append(
                    f"{attr.capitalize()}: '{target_row[attr]}' ≠ '{candidate_row[attr]}'"
                )
        # If one or both are empty, no penalty and no credit

    attr_score = matches / comparisons if comparisons > 0 else 0.5  # neutral when no data
    return attr_score, penalty, conflicts


def calculate_match(
    target_idx: int,
    df: pd.DataFrame,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """
    Score every other material against the target using:

        Final = (cosine_sim × 0.6) + (attr_score × 0.4) − penalty

    Parameters
    ----------
    target_idx : int   positional index of the target row in *df*
    df         : pd.DataFrame   the full catalogue
    embeddings : np.ndarray     shape (N, D) pre-computed embeddings

    Returns
    -------
    pd.DataFrame sorted descending by final_score, target row excluded.
    """
    target_emb = embeddings[target_idx].reshape(1, -1)
    cos_scores = cosine_similarity(target_emb, embeddings).flatten()

    target_row = df.iloc[target_idx]
    results = []

    for idx in range(len(df)):
        if idx == target_idx:
            continue

        candidate_row = df.iloc[idx]
        cos = float(cos_scores[idx])
        attr_score, penalty, conflicts = _attr_match_score(target_row, candidate_row)

        final_score = (cos * 0.6) + (attr_score * 0.4) - penalty

        results.append(
            {
                "rank": 0,  # filled after sort
                "id": candidate_row["id"],
                "cpse": candidate_row["cpse"],
                "legacy_code": candidate_row["legacy_code"],
                "raw_description": candidate_row["raw_description"],
                "noun": candidate_row["noun"],
                "material": candidate_row["material"],
                "grade": candidate_row["grade"],
                "size": candidate_row["size"],
                "cosine_sim": round(cos, 4),
                "attr_score": round(attr_score, 4),
                "penalty": round(penalty, 4),
                "final_score": round(final_score, 4),
                "conflicts": conflicts,
            }
        )

    result_df = pd.DataFrame(results).sort_values("final_score", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1
    return result_df
