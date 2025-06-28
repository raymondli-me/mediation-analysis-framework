# MASTER CONTEXT: PoMA Frameworks Comparison Project

## Project Mission
Systematically demonstrate the relationships between four mediation analysis frameworks:
1. Traditional Mediation (Baron & Kenny)
2. Frisch-Waugh-Lovell (FWL)
3. Double Machine Learning (DML)
4. Causal Mediation (Natural Direct/Indirect Effects)

## Key Thesis
- In linear settings: Traditional ≡ FWL ≡ DML ≡ Causal (no interaction)
- In non-linear settings: DML handles it, others fail
- With interactions: Only causal framework gives meaningful decomposition
- **The causal framework subsumes all others**

## Critical Files to Remember

### Data Generation
- `/src/data_generation.py` - 11 scenarios including linear, non-linear, interactions, edge cases
- Key class: `DataGenerator` with methods like `linear_mediation()`, `linear_interaction()`, `symmetric_ushaped()`

### Core Implementations (TO BUILD)
- `/src/traditional_mediation.py` - Baron & Kenny approach
- `/src/fwl_approach.py` - Frisch-Waugh-Lovell 
- `/src/dml_approach.py` - DML with reduction formula
- `/src/causal_mediation.py` - NDE/NIE estimation

### Demonstrations
- `/demonstrations/step1_linear_equivalence.py` - ✅ DONE: Shows perfect equivalence
- `/demonstrations/step2_nonlinear_dml.py` - TODO: DML robustness
- `/demonstrations/step3_interactions_causal.py` - TODO: Causal framework superiority
- `/demonstrations/comprehensive_comparison.py` - TODO: All methods, all scenarios

## Mathematical Formulas to Remember

### Traditional/FWL/DML (estimate same thing)
```
Total: Y = c*X + ε
Direct: Y = c'*X + b*M + ε
PoMA = 1 - (c'/c) = (c-c')/c
```

### DML Reduction Formula (from paper)
```
c'/c = [1 - Cov(Ŷ,X̂)/Cov(Y,X) - C1 - C2] / [1 - Var(X̂)/Var(X) - C3]
Where:
- Ŷ, X̂ are predictions from M
- C1, C2, C3 are correction terms
```

### Natural Effects (Causal Framework)
```
Total Effect (TE) = E[Y(1)] - E[Y(0)]
Natural Direct Effect (NDE) = E[Y(1,M(0))] - E[Y(0,M(0))]
Natural Indirect Effect (NIE) = E[Y(1,M(1))] - E[Y(1,M(0))]
TE = NDE + NIE (always!)
```

## Key Insights

1. **When X-Y correlation ≈ 0** (symmetric cases):
   - Traditional PoMA undefined (divide by zero)
   - DML formula unstable
   - But NDE/NIE still well-defined!

2. **With interactions (Y = γX + βM + δXM)**:
   - CDE depends on M level
   - NDE/NIE provide stable decomposition
   - Traditional approaches misleading

3. **DML's power**:
   - Uses ML to handle non-linearity in nuisance functions
   - But still estimates same CDE as traditional (just more robustly)
   - The reduction formula is a computational shortcut

## Implementation Strategy

### Phase 1: Core Modules (NOW)
1. Create `/src/traditional_mediation.py` with:
   - `estimate_effects()` - returns total, direct, indirect
   - `calculate_poma()` - handles edge cases
   
2. Create `/src/fwl_approach.py` with:
   - `fwl_regression()` - implements theorem
   - Shows it's identical to traditional

3. Create `/src/dml_approach.py` with:
   - `dml_mediation()` - cross-fitted estimation
   - `reduction_formula()` - analytical shortcut
   - Allows both linear and ML predictors

4. Create `/src/causal_mediation.py` with:
   - `estimate_nde()` - Natural Direct Effect
   - `estimate_nie()` - Natural Indirect Effect
   - Works even with interactions

### Phase 2: Demonstrations
1. Step 2: Non-linear M→Y relationship
   - Show traditional/FWL fail (biased)
   - DML succeeds (unbiased)
   - Reduction formula matches empirical

2. Step 3: X*M interaction
   - All estimate CDE but it's not meaningful
   - Only NDE/NIE give stable decomposition
   - Visualize effect heterogeneity

### Phase 3: Comprehensive Testing
- Run all methods on all 11 scenarios
- Create comparison matrix
- Identify when each method appropriate

## Code Patterns to Use

```python
# Standard structure for each method
def estimate_method_name(X, M, Y, **kwargs):
    """Docstring explaining method"""
    
    # Calculate total effect
    total_effect = ...
    
    # Calculate direct effect (method-specific)
    direct_effect = ...
    
    # Calculate indirect effect
    indirect_effect = total_effect - direct_effect
    
    # Calculate PoMA with stability check
    if abs(total_effect) > 1e-10:
        poma = indirect_effect / total_effect
    else:
        poma = np.nan
    
    return {
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'poma': poma,
        'method': 'method_name',
        'additional_info': ...
    }
```

## Testing Strategy
- Each method should pass linear equivalence test
- DML should handle non-linear cases
- Causal should handle interactions
- Edge cases should produce appropriate warnings

## Remember
- The goal is CLARITY over complexity
- Each demonstration should have an "aha!" moment
- Document WHY not just HOW
- Make it useful for practitioners

## Current Status: Ready to implement core modules!