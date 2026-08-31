# Baseline Reproduction Results

Environment:

- Python 3.12.8
- NumPy 2.0.2
- Seed: 0

| Model | Split | GAUC | nDCG@5 | Primary |
|---|---|---:|---:|---:|
| Random | Valid | 0.4990 | 0.4663 | 0.4827 |
| Random | Test | 0.4999 | 0.4514 | 0.4757 |
| Popularity | Valid | 0.6387 | 0.5227 | 0.5807 |
| Popularity | Test | 0.6308 | 0.5121 | 0.5715 |
| FM | Valid | 0.6671 | 0.5358 | 0.6015 |
| FM | Test | 0.6621 | 0.5286 | 0.5953 |

The official starter-kit baselines were successfully reproduced.