# Project Summary

## Overview

This project provides a comprehensive comparison of mediation analysis frameworks, addressing the critical issue of Percentage of Mediated Accuracy (PoMA) instability in symmetric non-linear relationships.

## Repository Information

- **GitHub URL**: https://github.com/raymondli-me/mediation-analysis-framework
- **License**: MIT
- **Language**: Python 3.8+

## Key Findings

1. **PoMA Instability**: Discovered that covariance-based PoMA calculations can produce nonsensical results (e.g., -35%, 1272%) when cov(Y,X) ≈ 0 in symmetric relationships like X² → M → sin(M).

2. **Framework Equivalence**: Proved that Traditional = FWL = DML (with linear models) when relationships are linear and no interactions exist.

3. **Non-linear Robustness**: DML with machine learning models handles non-linearity effectively, while traditional methods fail.

4. **Interaction Handling**: Only the Natural Effects framework properly handles treatment-mediator interactions.

## Main Components

### 1. Theoretical Document (`docs/mediation_theory.pdf`)
- Complete mathematical derivation of all frameworks
- Proofs of equivalences and conditions
- Explanation of PoMA instability
- 13-page comprehensive theory document

### 2. Core Implementations (`src/`)
- `traditional_mediation.py`: Baron & Kenny approach
- `fwl_approach.py`: Frisch-Waugh-Lovell theorem
- `dml_approach.py`: Double Machine Learning with reduction formula
- `causal_mediation.py`: Natural Direct/Indirect Effects
- `data_generation.py`: 11 test scenarios

### 3. Demonstrations (`demonstrations/`)
- Step 1: Linear equivalence proof
- Step 2: Non-linear robustness with DML
- Step 3: Interaction handling with Natural Effects
- Comprehensive comparison across all scenarios
- Edge cases analysis

### 4. Tools (`tools/`)
- `diagnostic_toolkit.py`: Analyze your data and get method recommendations

### 5. Interactive Interface
- `main.py`: Menu-driven system to access all functionality

## Quick Start

```bash
# Clone the repository
git clone https://github.com/raymondli-me/mediation-analysis-framework.git
cd mediation-analysis-framework

# Install dependencies
pip install -r requirements.txt

# Run interactive menu
python main.py
```

## Key Recommendations

1. **Always visualize your data** before conducting mediation analysis
2. **Check for near-zero correlations** - if |cor(X,Y)| < 0.05, PoMA will be unstable
3. **Test for non-linearity** - use DML with ML if detected
4. **Test for interactions** - use Natural Effects if present
5. **Report effect sizes**, not just proportions (PoMA)

## When to Use Each Method

| Scenario | Method | Reason |
|----------|--------|---------|
| Linear, no interaction | Traditional/FWL | Simple, proven |
| Non-linear relationships | DML with ML | Handles complexity |
| Treatment-mediator interaction | Natural Effects | Proper decomposition |
| Small sample (n < 200) | Traditional | Avoid overfitting |
| Near-zero effects | Any + Bootstrap CI | Need uncertainty |

## Citation

If you use this toolkit in your research:

```bibtex
@software{mediation_analysis_framework,
  title = {Mediation Analysis Framework Comparison},
  author = {Mediation Analysis Project},
  year = {2024},
  url = {https://github.com/raymondli-me/mediation-analysis-framework}
}
```

## Contact

For questions or issues, please open an issue on GitHub.