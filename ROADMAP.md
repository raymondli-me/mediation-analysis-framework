# PoMA Frameworks Comparison - Roadmap

## Current Status

✅ **Completed:**
- Project structure and planning
- Data generation module with 11 scenarios
- Step 1 demonstration: Linear equivalence

## Next Steps

### Immediate (This Week)

1. **Complete Core Framework Implementations**
   - [ ] `src/traditional_mediation.py` - Traditional Baron & Kenny approach
   - [ ] `src/fwl_approach.py` - Frisch-Waugh-Lovell implementation
   - [ ] `src/dml_approach.py` - Double Machine Learning with formula
   - [ ] `src/causal_mediation.py` - Natural Direct/Indirect Effects

2. **Step 2 Demonstration: Non-linear Robustness**
   - [ ] Show how traditional/FWL fail with non-linear M→Y
   - [ ] Demonstrate DML's robustness
   - [ ] Validate DML reduction formula

3. **Step 3 Demonstration: Interactions**
   - [ ] Show limitations of CDE with interactions
   - [ ] Implement NDE/NIE estimation
   - [ ] Compare all approaches

### Medium Term (Next 2 Weeks)

4. **Edge Cases and Diagnostics**
   - [ ] Symmetric relationships (U-shaped)
   - [ ] Suppression effects
   - [ ] Near-zero total effects
   - [ ] Diagnostic toolkit

5. **Comprehensive Comparison**
   - [ ] Run all methods on all scenarios
   - [ ] Create comparison matrix
   - [ ] Performance benchmarks

6. **Documentation**
   - [ ] Mathematical proofs notebook
   - [ ] User guide
   - [ ] API documentation

### Long Term (Month 2)

7. **Extensions**
   - [ ] Multiple mediators
   - [ ] Continuous treatments
   - [ ] Sensitivity analysis
   - [ ] Confidence intervals

8. **Applications**
   - [ ] Real data examples
   - [ ] Integration with existing packages
   - [ ] Streamlit dashboard

## Key Insights So Far

1. **Linear Case**: Perfect equivalence between Traditional = FWL = DML
2. **Non-linear Case**: DML handles it, others don't
3. **Interaction Case**: Only causal framework meaningful
4. **Computational**: DML provides the engine, causal provides the theory

## Technical Decisions

- Using scikit-learn for linear models
- XGBoost for non-linear ML
- May add EconML for causal methods
- Focusing on clarity over performance

## Open Questions

1. How to best estimate NDE/NIE with continuous treatments?
2. Should we implement sensitivity analysis?
3. How to handle multiple mediators elegantly?
4. Best visualization strategy for interactions?

## Success Criteria

- [ ] Clear demonstration of when methods agree/disagree
- [ ] Practical guidance for practitioners
- [ ] Well-tested, documented code
- [ ] At least one "aha!" moment per demonstration

## Repository Structure

```
Current state:
✅ README.md
✅ requirements.txt
✅ project_plan.md
✅ src/
   ✅ __init__.py
   ✅ data_generation.py
   ⏳ traditional_mediation.py
   ⏳ fwl_approach.py
   ⏳ dml_approach.py
   ⏳ causal_mediation.py
✅ demonstrations/
   ✅ step1_linear_equivalence.py
   ⏳ step2_nonlinear_dml.py
   ⏳ step3_interactions_causal.py
   ⏳ comprehensive_comparison.py
```

## How to Contribute

1. Pick a task from "Next Steps"
2. Create implementation
3. Add tests
4. Update documentation
5. Run demonstrations

## Contact

Questions? Ideas? Found a bug? Create an issue!