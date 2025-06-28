# Project Implementation Plan

## Phase 1: Foundation (Week 1)

### 1.1 Core Infrastructure
- [ ] Set up project structure and dependencies
- [ ] Create data generation module with scenarios:
  - Linear relationships (baseline)
  - Non-linear relationships (polynomial, exponential)
  - Interaction effects
  - Threshold/saturation effects
  - Symmetric relationships (U-shaped)

### 1.2 Traditional Approaches
- [ ] Implement traditional mediation (Baron & Kenny)
- [ ] Implement FWL approach
- [ ] Create visualization utilities
- [ ] Unit tests for correctness

## Phase 2: Advanced Methods (Week 2)

### 2.1 DML Implementation
- [ ] Port DML mediation from original code
- [ ] Implement DML reduction formula
- [ ] Add cross-fitting variants
- [ ] Compare with sklearn's causal inference tools

### 2.2 Causal Mediation Framework
- [ ] Implement Natural Direct Effect (NDE) estimation
- [ ] Implement Natural Indirect Effect (NIE) estimation
- [ ] Handle continuous and binary treatments
- [ ] Bootstrap confidence intervals

## Phase 3: Systematic Comparison (Week 3)

### 3.1 Step 1: Linear Equivalence
**Goal**: Prove Traditional ≡ FWL ≡ DML in linear settings

- [ ] Generate perfectly linear data
- [ ] Run all four methods
- [ ] Show numerical equivalence (within machine precision)
- [ ] Visualize the relationships
- [ ] Document theoretical explanation

### 3.2 Step 2: Non-linear Robustness
**Goal**: Show DML handles non-linearity while others fail

- [ ] Generate data with:
  - X → M: linear
  - M → Y: non-linear (polynomial, sine, exponential)
  - No interaction
- [ ] Compare performance:
  - Traditional: biased
  - FWL: biased
  - DML: unbiased
  - Causal: matches DML
- [ ] Show DML reduction formula accuracy

### 3.3 Step 3: Interaction Complexity
**Goal**: Demonstrate only causal framework handles interactions properly

- [ ] Generate data with X*M interaction
- [ ] Show:
  - Traditional/FWL/DML give "controlled direct effect"
  - This varies with M level
  - NDE/NIE provide meaningful decomposition
- [ ] Visualize how effect changes with mediator level

## Phase 4: Edge Cases and Stability (Week 4)

### 4.1 Problematic Scenarios
- [ ] Near-zero total effects (symmetric relationships)
- [ ] Perfect mediation
- [ ] Suppression effects
- [ ] Multiple mediators

### 4.2 Diagnostic Tools
- [ ] Linearity tests
- [ ] Interaction detection
- [ ] Stability assessment
- [ ] Recommendation engine

## Phase 5: Documentation and Dissemination

### 5.1 Comprehensive Report
- [ ] Mathematical proofs of equivalence
- [ ] Empirical validation results
- [ ] Practical guidelines
- [ ] Software documentation

### 5.2 Interactive Demonstrations
- [ ] Jupyter notebooks with widgets
- [ ] Streamlit app for exploration
- [ ] Tutorial videos/animations

## Key Demonstrations to Build

### Demo 1: "The Three Amigos" (Linear Case)
```python
# Show perfect agreement between:
# 1. Traditional: Y ~ β₀ + β₁X + β₂M
# 2. FWL: e_Y ~ β₁e_X (after partialing out M)
# 3. DML: Cross-fitted version of FWL
# All should give identical β₁ (controlled direct effect)
```

### Demo 2: "DML Saves the Day" (Non-linear Case)
```python
# M = f(X) + ε (linear)
# Y = g(M) + h(X) + ε (g is non-linear)
# Show:
# - Traditional/FWL: biased due to misspecification
# - DML: correctly estimates h'(X) using ML for g(M)
# - DML reduction formula matches empirical DML
```

### Demo 3: "Only Causal Survives" (Interaction Case)
```python
# Y = α + βX + γM + δXM + ε
# Show:
# - All methods estimate CDE, but it depends on M
# - Only NDE/NIE give stable decomposition
# - Visualize how direct effect changes with M
```

### Demo 4: "When Everything Breaks" (Symmetric Case)
```python
# X ~ Uniform(-3, 3)
# M = X²
# Y = sin(M)
# Show:
# - cov(X,Y) ≈ 0
# - Traditional PoMA: undefined
# - DML PoMA: unstable
# - NDE/NIE: still meaningful
```

## Success Metrics

1. **Theoretical Clarity**: Clear proofs of when methods are equivalent
2. **Empirical Validation**: Simulations confirming theoretical results
3. **Practical Utility**: Guidelines for practitioners
4. **Code Quality**: Well-tested, documented, reusable code
5. **Educational Value**: Clear explanations and visualizations

## Timeline

- Week 1: Foundation and traditional methods
- Week 2: Advanced methods (DML, causal)
- Week 3: Systematic comparisons
- Week 4: Edge cases and diagnostics
- Week 5: Documentation and polish

## Next Steps

1. Create requirements.txt with dependencies
2. Set up data generation module
3. Implement traditional mediation analysis
4. Begin step 1 demonstration