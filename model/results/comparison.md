# Model comparison (time-based hold-out)

Train rows: 5040 | Test rows: 2160 | Test theft rate: 3.9% (a model flagging at random scores PR-AUC ~ 0.04)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit (s) |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.941 | 0.565 | 0.706 | 0.982 | 0.782 | 0.18 |
| Local Outlier Factor | 0.177 | 0.259 | 0.211 | 0.622 | 0.101 | 0.02 |
| One-Class SVM | 0.512 | 0.518 | 0.515 | 0.569 | 0.524 | 0.04 |
| Elliptic Envelope | 0.703 | 0.529 | 0.604 | 0.973 | 0.706 | 0.43 |
| Z-score rule (baseline) | 0.316 | 0.882 | 0.466 | 0.977 | 0.758 | 0.00 |

## Robustness over 5 independent simulated datasets (mean +/- std)

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Isolation Forest | 0.609 +/- 0.233 | 0.970 +/- 0.019 | 0.792 +/- 0.112 |
| Local Outlier Factor | 0.242 +/- 0.226 | 0.602 +/- 0.184 | 0.217 +/- 0.241 |
| One-Class SVM | 0.498 +/- 0.167 | 0.596 +/- 0.194 | 0.459 +/- 0.230 |
| Elliptic Envelope | 0.627 +/- 0.144 | 0.969 +/- 0.015 | 0.784 +/- 0.126 |
| Z-score rule (baseline) | 0.574 +/- 0.095 | 0.924 +/- 0.063 | 0.674 +/- 0.171 |

## Isolation Forest: contamination sensitivity

| contamination | precision | recall | F1 |
|---|---|---|---|
| 0.01 | 1.000 | 0.271 | 0.426 |
| 0.02 | 1.000 | 0.506 | 0.672 |
| 0.05 | 0.941 | 0.565 | 0.706 |
| 0.08 | 0.492 | 0.741 | 0.592 |
| 0.10 | 0.368 | 0.906 | 0.524 |
| 0.15 | 0.226 | 1.000 | 0.369 |

## Recall by theft type (fraction of each theft type caught)

| Model | abnormal_surge | meter_bypass | meter_stall |
|---|---|---|---|
| Isolation Forest | 1.00 | 0.00 | 1.00 |
| Local Outlier Factor | 0.10 | 0.00 | 0.55 |
| One-Class SVM | 1.00 | 0.03 | 0.87 |
| Elliptic Envelope | 0.10 | 0.16 | 1.00 |
| Z-score rule (baseline) | 1.00 | 0.73 | 1.00 |

_All numbers are from simulated data and describe this simulator, not real utility data._
