# data/raw/

Place the raw MTSamples dataset here before running preprocessing.

## Required file

`mtsamples.csv`

Download from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Expected columns after download:
- (unnamed index)
- description
- medical_specialty
- sample_name
- transcription
- keywords

Expected size: ~17 MB, 4,999 rows, 40 unique medical specialties.

## Datasets NOT used in this project

The following datasets were evaluated and excluded:

| Dataset | Reason |
|---------|--------|
| BioASQ Task B (factoid QA) | Wrong task: extractive QA, not specialty classification |
| MIMIC-III demo (archive/) | Too small (100 patients), no free-text clinical notes |

## Optional: PMC-Patients (for pretraining only)

Download with HuggingFace CLI (Safe Path only — no label derivation):

```bash
huggingface-cli download zhengyun21/PMC-Patients --repo-type dataset
```

Reference: Zhengyun et al., Nature Scientific Data 2023.
License: CC BY 4.0.
