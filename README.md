# OmniMat: CPSE Material Intelligence 🔩

**AI-Driven Harmonization of Material Codes across Central Public Sector Enterprises (CPSEs)**  
*Smart India Hackathon 2026 — Problem Statement 26099*

---

## 🎯 Overview
In industrial procurement across Indian CPSEs (ONGC, IOCL, BPCL, HPCL, GAIL, etc.), items with identical physical and engineering specifications often carry completely divergent material descriptions and legacy catalog codes. Conversely, materials that appear 95% textually similar might differ in a critical safety or engineering attribute (such as metallurgical grade or pipe schedule), making accidental auto-merging catastrophic.

**OmniMat** solves this using an **Attribute-Aware Conflict Penalty AI Engine**:
- **Dense Semantic Matching:** Encodes raw catalog descriptions into semantic embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Deterministic Attribute Resolution:** Extracts and cross-validates critical engineering attributes (`noun`, `material`, `grade`, `size`).
- **Attribute-Aware Conflict Penalty (Hard Veto):** If semantic similarity is high but critical engineering parameters conflict (e.g., Grade 304 vs 316), a hard penalty is deducted, immediately preventing false merges.

$$\text{Final Score} = (\text{Cosine Similarity} \times 0.6) + (\text{Attribute Match} \times 0.4) - \text{Penalty}$$

---

## 🚀 Key Features
- **Zero-Latency Architecture:** Fully self-contained without live database or third-party API dependencies.
- **Enterprise Dashboard:** Real-time KPI metrics, CPSE breakdown, and catalog exploration.
- **Harmonization Engine:** Side-by-side entity resolution demo with automatic confidence verdicts:
  - 🟢 **High Confidence Equivalent (> 0.85):** Safe for auto-merge.
  - 🔴 **Critical Conflict Detected (< 0):** Hard veto triggered; false merge prevented.
  - 🟡 **Possible Match (0.50 – 0.85):** Flagged for manual review.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit**
- **Pandas**
- **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- **Scikit-Learn**

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dorodb-web22/OmniMat-1.git
   cd OmniMat-1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment
Ready for 1-click deployment on **Streamlit Community Cloud**:
- **Main file path:** `app.py`
- No environment variables or external API keys needed.
