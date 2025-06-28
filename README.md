# Mediation Analysis Frameworks Comparison

A comprehensive Python toolkit for understanding and comparing different mediation analysis frameworks, with a focus on the Percentage of Mediated Accuracy (PoMA) calculation and its stability across methods.

## Overview

This project emerged from discovering that covariance-based PoMA calculations can produce nonsensical results (e.g., -35%, 1272%) in symmetric non-linear relationships. It provides a systematic comparison of four major mediation analysis frameworks:

1. **Traditional Mediation Analysis** (Baron & Kenny) - The classical approach
2. **Frisch-Waugh-Lovell (FWL) Approach** - Residualization-based approach
3. **Double Machine Learning (DML) Approach** - Modern ML-based approach with cross-fitting
4. **Causal Mediation Framework** (Natural Direct/Indirect Effects) - Handles interactions properly

## Key Questions Addressed

1. When are these approaches equivalent?
2. How does the causal framework subsume the others?
3. What happens under different conditions (linearity, interactions, etc.)?
4. How does the DML reduction formula relate to causal quantities?
5. Why does PoMA become unstable in symmetric relationships?
6. When should practitioners use each method?

## Project Structure

```
poma_frameworks_comparison/
├── README.md                          # This file
├── MASTER_CONTEXT.md                  # Master planning document
├── requirements.txt                   # Dependencies
├── main.py                           # Interactive menu system
│
├── src/                              # Core implementation modules
│   ├── __init__.py
│   ├── data_generation.py           # 11 data scenarios (linear, nonlinear, interactions, edge cases)
│   ├── traditional_mediation.py     # Baron & Kenny approach with bootstrap CI
│   ├── fwl_approach.py             # Frisch-Waugh-Lovell implementation
│   ├── dml_approach.py             # Double ML with reduction formula
│   ├── causal_mediation.py         # Natural Direct/Indirect Effects
│   └── unified_framework.py        # Unified interface for all methods
│
├── demonstrations/                   # Step-by-step demonstrations
│   ├── step1_linear_equivalence.py  # Show Traditional = FWL = DML in linear case
│   ├── step2_nonlinear_dml.py      # DML handles non-linearity, others fail
│   ├── step3_interactions_causal.py # Only causal framework handles interactions
│   ├── comprehensive_comparison.py   # All methods on all 11 scenarios
│   └── edge_cases_demo.py          # Symmetric relationships, suppression, etc.
│
├── tools/                           # Practical tools
│   └── diagnostic_toolkit.py        # Analyze your data, get recommendations
│
├── notebooks/                        # Educational content
│   └── mathematical_proofs.py       # Proofs of equivalences and formulas
│
└── outputs/                         # Results directory (auto-created)
    ├── figures/                     # All visualizations
    └── tables/                      # Numerical results
```

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the interactive main menu
python main.py

# Or run specific demonstrations
python demonstrations/step1_linear_equivalence.py
python demonstrations/step2_nonlinear_dml.py
python demonstrations/step3_interactions_causal.py
python demonstrations/comprehensive_comparison.py
python demonstrations/edge_cases_demo.py
```

## Key Findings

1. **Linear World**: Traditional ≡ FWL ≡ DML (with linear models) - all give identical results
2. **Non-linear World**: DML with ML models handles non-linearity; linear methods are biased
3. **Interaction World**: Only causal framework (Natural Effects) gives meaningful decomposition
4. **Edge Cases**: PoMA becomes extremely unstable when cov(Y,X) ≈ 0
5. **Suppression**: PoMA > 100% is possible and meaningful when effects have opposite signs

## Usage Examples

### Basic Analysis
```python
from src.traditional_mediation import estimate_effects

# Your data
X = treatment_variable
M = mediator_variable  
Y = outcome_variable

# Traditional analysis
results = estimate_effects(X, M, Y)
print(f"Direct effect: {results['direct_effect']:.3f}")
print(f"Indirect effect: {results['indirect_effect']:.3f}")
print(f"PoMA: {results['poma']:.1%}")
```

### Diagnostic Toolkit
```python
from tools.diagnostic_toolkit import diagnose_mediation_data

# Analyze your data and get recommendations
report = diagnose_mediation_data(X, M, Y)
report.show()
```

### Compare All Methods
```python
from src.unified_framework import run_all_methods
from sklearn.ensemble import RandomForestRegressor

# Run all methods
results = run_all_methods(
    X, M, Y,
    ml_models={'y': RandomForestRegressor(), 'x': RandomForestRegressor()}
)
print(results)
```

## When to Use Each Method

| Scenario | Recommended Method | Why |
|----------|-------------------|-----|
| Linear relationships, no interaction | Traditional/FWL | Simple, fast, proven |
| Non-linear relationships | DML with ML | Handles complex patterns |
| Treatment-mediator interaction | Natural Effects | Proper causal decomposition |
| Small sample (n < 200) | Traditional | Avoid overfitting |
| Near-zero effects | Any + Bootstrap CI | Need uncertainty quantification |

## PoMA Stability Issues

The project reveals critical issues with PoMA calculations:

1. **Symmetric Relationships**: When X² → M → sin(M), cov(Y,X) ≈ 0, leading to PoMA = -35% or 1272%
2. **Near-Zero Total Effect**: Small denominators cause numerical explosion
3. **Suppression Effects**: Direct and indirect effects with opposite signs give PoMA > 100%

**Recommendation**: Always report effect sizes alongside PoMA, and be cautious in edge cases.

## Mathematical Foundations

The `notebooks/mathematical_proofs.py` file contains proofs showing:
- Traditional ≡ FWL (Frisch-Waugh-Lovell theorem)
- DML reduction formula: c'/c = f(predictions)
- Natural effects decomposition: Total = NDE + NIE
- CDE = NDE when no interaction

## References

- Baron, R. M., & Kenny, D. A. (1986). The moderator-mediator variable distinction
- Pearl, J. (2001). Direct and indirect effects
- Chernozhukov et al. (2018). Double/debiased machine learning
- VanderWeele, T. J. (2015). Explanation in causal inference
- Imai et al. (2010). A general approach to causal mediation analysis

## Contributing

Contributions welcome! The project is designed to be extensible:
- Add new methods to `src/`
- Add new scenarios to `data_generation.py`
- Add new demonstrations to `demonstrations/`

## License

MIT License - see LICENSE file for details.