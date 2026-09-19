# Denial Root Cause & Recovery Intelligence

An end-to-end ML + multi-agent system for healthcare claim denial management — classifies denied claims by true root cause, predicts recoverability, and prescribes the optimal next-best action (Appeal / Resubmit / Write Off / Bill Patient) with expected-value justification.

Built as part of a 3-component Denials Management program.

---

## Problem

Healthcare claims get denied at high rates for reasons ranging from missing prior authorization to medical necessity disputes. Revenue cycle teams triage these manually — figuring out *why* a claim was denied, whether it's worth pursuing, and what action to take — claim by claim, relying on individual analyst experience rather than systematic data.

## Solution

Two pipelines working together:

### 1. ML Modeling Pipeline (`Recovery_Intelligence_Modeling_v2/`)

11-stage pipeline that turns raw EDI claims data (837/277/835) into a trained recovery-prediction model:

| Stage | What It Does |
|---|---|
| Data Understanding | Automated EDA via ydata-profiling |
| Data Merging | 837 + 277 + 835 EDI join with outcome derivation |
| Pre-Split Cleaning | Dedup, 85% null threshold, type coercion |
| Stratified Split | Train/test with class-proportion preservation |
| Feature Engineering | 17 claim features + 12 payer-behavior detectors (CUSUM, PSI, OLS, Welch, HHI) + 3 early-warning signals |
| Post-Split Preprocessing | Imputation, encoding, scaling — fit on train only |
| Modeling | 13 model configs, Catch Rate Gate + F1 + CV stability ranking |

**Key Results:**
- Champion model catches **91% of denied claims** (466/512) in the test set
- ROC-AUC: **0.941** with cross-validation F1 stability at 0.781
- Top feature: `denial_count` (SHAP 0.3546), followed by `has_rarc` (0.0629)

### 2. Agent Pipeline (`Denial_Root_Cause_Recovery_Agents_v2/`)

LangGraph-based multi-agent system with 4 LLM sub-agents + 4 coded tools:

| Agent | Role |
|---|---|
| Root Cause Analyst | Classifies denial beyond the CARC code to true upstream origin |
| Evidence Analyst | Gathers supporting documentation for appeals |
| NBA Advisor | Prescribes next-best action with EV-backed justification |
| CARC/RARC Retrieval | Enriches denial codes with descriptions and context |

**Architecture:** CO/PR denials resolved by rule (contractual write-off / bill patient). OA/PI denials scored by ML model → recovery probability → expected value by action (recovery odds × claim amount − pursuit cost) → economically optimal recommendation.

## Tech Stack

- **ML:** Python, scikit-learn, LightGBM, XGBoost, CatBoost, SHAP, imbalanced-learn (SMOTE)
- **Statistical Detection:** scipy (CUSUM z-test, Welch t-test, PSI, OLS regression, HHI permutation test)
- **Agents:** LangGraph, LangChain, Anthropic Claude API
- **Vector Search:** FAISS, sentence-transformers (3-tier fallback: FAISS → numpy → metadata)
- **Data:** pandas, openpyxl, ydata-profiling
- **Deployment:** NeuroSAN AI platform (agent orchestration)

## Setup

### ML Pipeline
```bash
cd Recovery_Intelligence_Modeling_v2
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
python main.py
```

### Agent Pipeline
```bash
cd Denial_Root_Cause_Recovery_Agents_v2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create .env with: ANTHROPIC_API_KEY=your-key-here
python main.py
# Enter claim IDs when prompted, e.g.: PCN0000018, PCN0000003
```

## Project Structure

```
├── Recovery_Intelligence_Modeling_v2/
│   ├── main.py                    # Pipeline entry point
│   ├── config/                    # All configuration files
│   ├── src/                       # 11 numbered pipeline stages
│   ├── data/                      # Raw EDI input
│   └── output/                    # Model outputs, reports, logs
│
├── Denial_Root_Cause_Recovery_Agents_v2/
│   ├── main.py                    # Agent entry point
│   ├── Agents/                    # Agent definitions
│   ├── Config/                    # LLM and pipeline config
│   ├── Models/                    # LLM provider abstraction
│   ├── Graph/                     # LangGraph state machine
│   ├── Prompts/                   # System and user prompts
│   └── Data/                      # ADS + trained model
│
└── requirements.txt               # Master dependency manifest
```

## License

Proprietary — All rights reserved. (Set your own license terms here.)
