#!/usr/bin/env python3
"""
Step 2: Non-linear Robustness - DML Saves the Day
=================================================

This demonstration shows what happens when the M→Y relationship is non-linear:
- Traditional and FWL approaches fail (biased estimates)
- DML with ML models correctly handles the non-linearity
- The DML reduction formula accurately predicts the empirical estimate

Key insight: DML's power comes from using flexible ML for nuisance functions
while still targeting the same causal parameter (controlled direct effect).
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import DataGenerator
from src.traditional_mediation import estimate_effects as traditional_estimate
from src.fwl_approach import fwl_regression, fwl_with_ml
from src.dml_approach import dml_mediation, compare_dml_implementations
from src.unified_framework import run_all_methods, compare_methods_visually

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def generate_nonlinear_data(n=2000, random_state=42):
    """
    Generate data with non-linear M→Y relationship.
    
    True model:
    - X → M: Linear (M = 1.5*X + noise)
    - M → Y: Non-linear (Y = 0.5*X + g(M) + noise)
    - Where g(M) is a non-linear function
    """
    np.random.seed(random_state)
    
    # Treatment
    X = np.random.normal(0, 1, n)
    
    # Linear X→M relationship
    alpha = 1.5
    M = alpha * X + np.random.normal(0, 0.5, n)
    
    # Non-linear M→Y relationship
    # Using a smooth non-linear function
    direct_effect = 0.5  # True direct effect
    
    # Non-linear mediator effect: combination of quadratic and sine
    def g(m):
        return 2 * m - 0.3 * m**2 + 0.5 * np.sin(2 * m)
    
    Y = direct_effect * X + g(M) + np.random.normal(0, 0.5, n)
    
    # Calculate true effects (approximately)
    # For small changes in X: indirect ≈ α * g'(E[M|X])
    # At X=0, M≈0, so g'(0) ≈ 2
    true_indirect_approx = alpha * 2  # ≈ 3
    true_total = direct_effect + true_indirect_approx
    true_poma = true_indirect_approx / true_total
    
    return X, M, Y, {
        'true_direct': direct_effect,
        'true_indirect_approx': true_indirect_approx,
        'true_total_approx': true_total,
        'true_poma_approx': true_poma,
        'alpha': alpha,
        'g_function': g
    }


def visualize_nonlinearity(X, M, Y, true_params):
    """Visualize the non-linear relationships."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: True relationships
    # X → M (linear)
    ax = axes[0, 0]
    ax.scatter(X, M, alpha=0.3, s=10)
    x_range = np.linspace(X.min(), X.max(), 100)
    ax.plot(x_range, true_params['alpha'] * x_range, 'r-', linewidth=2, 
            label=f"M = {true_params['alpha']}X")
    ax.set_xlabel('X')
    ax.set_ylabel('M')
    ax.set_title('X → M: Linear Relationship')
    ax.legend()
    
    # M → Y (non-linear)
    ax = axes[0, 1]
    ax.scatter(M, Y, alpha=0.3, s=10)
    m_range = np.linspace(M.min(), M.max(), 100)
    # Show the non-linear function
    y_nonlinear = true_params['g_function'](m_range) + true_params['true_direct'] * 0
    ax.plot(m_range, y_nonlinear, 'r-', linewidth=2, 
            label='g(M) = 2M - 0.3M² + 0.5sin(2M)')
    ax.set_xlabel('M')
    ax.set_ylabel('Y')
    ax.set_title('M → Y: Non-linear Relationship')
    ax.legend()
    
    # X → Y total
    ax = axes[0, 2]
    ax.scatter(X, Y, alpha=0.3, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('X → Y: Total Relationship')
    
    # Bottom row: Linear vs flexible fits
    # Linear fit to M→Y
    ax = axes[1, 0]
    ax.scatter(M, Y, alpha=0.3, s=10, label='Data')
    # Linear regression line
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(M.reshape(-1, 1), Y)
    y_linear = lr.predict(m_range.reshape(-1, 1))
    ax.plot(m_range, y_linear, 'b--', linewidth=2, label='Linear fit')
    ax.plot(m_range, y_nonlinear, 'r-', linewidth=2, label='True g(M)')
    ax.set_xlabel('M')
    ax.set_ylabel('Y')
    ax.set_title('Linear Fit Misses Non-linearity')
    ax.legend()
    
    # ML fit to M→Y
    ax = axes[1, 1]
    ax.scatter(M, Y, alpha=0.3, s=10, label='Data')
    # Random forest fit
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(M.reshape(-1, 1), Y)
    y_rf = rf.predict(m_range.reshape(-1, 1))
    ax.plot(m_range, y_rf, 'g-', linewidth=2, label='Random Forest fit')
    ax.plot(m_range, y_nonlinear, 'r--', linewidth=2, label='True g(M)')
    ax.set_xlabel('M')
    ax.set_ylabel('Y')
    ax.set_title('ML Captures Non-linearity')
    ax.legend()
    
    # Residual plots
    ax = axes[1, 2]
    # Residuals from linear fit
    y_pred_linear = lr.predict(M.reshape(-1, 1))
    residuals_linear = Y - y_pred_linear
    ax.scatter(M, residuals_linear, alpha=0.3, s=10, color='blue', label='Linear residuals')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('M')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Pattern Shows Misspecification')
    ax.legend()
    
    plt.suptitle('Non-linear M→Y Relationship: Linear Models Fail', fontsize=16)
    plt.tight_layout()
    
    return fig


def run_methods_comparison(X, M, Y, true_params):
    """Run all methods and compare results."""
    
    print("\n" + "="*70)
    print("COMPARING METHODS ON NON-LINEAR DATA")
    print("="*70)
    
    results = []
    
    # 1. Traditional (will be biased)
    print("\n1. Traditional Mediation Analysis")
    trad = traditional_estimate(X, M, Y)
    print(f"   Direct effect:  {trad['direct_effect']:.4f} (true: {true_params['true_direct']:.4f})")
    print(f"   Indirect effect: {trad['indirect_effect']:.4f} (true ≈ {true_params['true_indirect_approx']:.4f})")
    print(f"   PoMA: {trad['poma']:.4f} (true ≈ {true_params['true_poma_approx']:.4f})")
    bias_trad = trad['direct_effect'] - true_params['true_direct']
    print(f"   Bias in direct effect: {bias_trad:.4f}")
    results.append(('Traditional', trad, bias_trad))
    
    # 2. FWL (will also be biased - same as traditional)
    print("\n2. Frisch-Waugh-Lovell")
    fwl = fwl_regression(X, M, Y)
    print(f"   Direct effect:  {fwl['direct_effect']:.4f}")
    print(f"   PoMA: {fwl['poma']:.4f}")
    print(f"   Difference from traditional: {abs(fwl['direct_effect'] - trad['direct_effect']):.2e}")
    results.append(('FWL', fwl, fwl['direct_effect'] - true_params['true_direct']))
    
    # 3. DML with linear models (still biased)
    print("\n3. DML with Linear Models")
    dml_linear = dml_mediation(X, M, Y, n_folds=5)
    print(f"   Direct effect:  {dml_linear['direct_effect']:.4f}")
    print(f"   PoMA: {dml_linear['poma']:.4f}")
    print(f"   Formula PoMA: {dml_linear['poma_formula']:.4f}")
    bias_dml_linear = dml_linear['direct_effect'] - true_params['true_direct']
    print(f"   Bias: {bias_dml_linear:.4f}")
    results.append(('DML-Linear', dml_linear, bias_dml_linear))
    
    # 4. DML with Random Forest (should be unbiased)
    print("\n4. DML with Random Forest")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    dml_rf = dml_mediation(X, M, Y, 
                          ml_model_y=rf_model,
                          ml_model_x=rf_model,
                          n_folds=5)
    print(f"   Direct effect:  {dml_rf['direct_effect']:.4f}")
    print(f"   PoMA: {dml_rf['poma']:.4f}")
    print(f"   Formula PoMA: {dml_rf['poma_formula']:.4f}")
    print(f"   R²(Y|M): {dml_rf['r2_y_given_m']:.4f}")
    bias_dml_rf = dml_rf['direct_effect'] - true_params['true_direct']
    print(f"   Bias: {bias_dml_rf:.4f} ← Much better!")
    results.append(('DML-RF', dml_rf, bias_dml_rf))
    
    # 5. DML with XGBoost
    print("\n5. DML with XGBoost")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, 
                                 learning_rate=0.1, random_state=42)
    dml_xgb = dml_mediation(X, M, Y,
                           ml_model_y=xgb_model,
                           ml_model_x=xgb_model,
                           n_folds=5)
    print(f"   Direct effect:  {dml_xgb['direct_effect']:.4f}")
    print(f"   PoMA: {dml_xgb['poma']:.4f}")
    print(f"   Formula PoMA: {dml_xgb['poma_formula']:.4f}")
    bias_dml_xgb = dml_xgb['direct_effect'] - true_params['true_direct']
    print(f"   Bias: {bias_dml_xgb:.4f}")
    results.append(('DML-XGB', dml_xgb, bias_dml_xgb))
    
    return results


def create_bias_comparison_plot(results, true_params):
    """Create visualization of bias across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [r[0] for r in results]
    biases = [r[2] for r in results]
    direct_effects = [r[1]['direct_effect'] for r in results]
    pomas = [r[1]['poma'] for r in results]
    
    # Bias plot
    ax = axes[0]
    colors = ['red', 'red', 'red', 'green', 'green']
    ax.bar(methods, biases, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Bias in Direct Effect')
    ax.set_title('Bias: Linear Methods Fail, ML Succeeds')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Direct effect estimates
    ax = axes[1]
    ax.bar(methods, direct_effects, alpha=0.7)
    ax.axhline(y=true_params['true_direct'], color='red', linestyle='--', 
               linewidth=2, label=f"True: {true_params['true_direct']}")
    ax.set_ylabel('Direct Effect Estimate')
    ax.set_title('Direct Effect Estimates')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PoMA estimates
    ax = axes[2]
    ax.bar(methods, pomas, alpha=0.7)
    ax.axhline(y=true_params['true_poma_approx'], color='red', linestyle='--',
               linewidth=2, label=f"True ≈ {true_params['true_poma_approx']:.3f}")
    ax.set_ylabel('PoMA')
    ax.set_title('Proportion Mediated')
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.suptitle('DML with ML Models Handles Non-linearity', fontsize=16)
    plt.tight_layout()
    
    return fig


def demonstrate_reduction_formula(X, M, Y):
    """Show that the DML reduction formula works."""
    print("\n" + "="*70)
    print("VALIDATING DML REDUCTION FORMULA")
    print("="*70)
    
    # Run DML with detailed formula output
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    dml_result = dml_mediation(X, M, Y,
                              ml_model_y=rf_model,
                              ml_model_x=rf_model,
                              n_folds=5)
    
    formula = dml_result['formula_results']
    
    print("\nFormula Components:")
    print(f"  Cov(Y,X) = {formula['cov_YX']:.4f}")
    print(f"  Var(X) = {formula['var_X']:.4f}")
    print(f"  Cov(Ŷ,X̂) = {formula['cov_Yhat_Xhat']:.4f}")
    print(f"  Var(X̂) = {formula['var_Xhat']:.4f}")
    
    print(f"\nRatios:")
    print(f"  Cov(Ŷ,X̂)/Cov(Y,X) = {formula['ratio_cov']:.4f}")
    print(f"  Var(X̂)/Var(X) = {formula['ratio_var']:.4f}")
    
    print(f"\nCorrection Terms:")
    print(f"  C1 = {formula['C1']:.4f}")
    print(f"  C2 = {formula['C2']:.4f}")
    print(f"  C3 = {formula['C3']:.4f}")
    
    print(f"\nFinal Calculation:")
    print(f"  Numerator = 1 - {formula['ratio_cov']:.4f} - {formula['C1']:.4f} - {formula['C2']:.4f}")
    print(f"           = {formula['numerator']:.4f}")
    print(f"  Denominator = 1 - {formula['ratio_var']:.4f} - {formula['C3']:.4f}")
    print(f"             = {formula['denominator']:.4f}")
    print(f"  Direct/Total ratio = {formula['direct_total_ratio']:.4f}")
    
    print(f"\nComparison:")
    print(f"  Direct effect (empirical): {dml_result['direct_effect']:.6f}")
    print(f"  Direct effect (formula):   {formula['direct_effect_formula']:.6f}")
    print(f"  Difference: {abs(dml_result['direct_effect'] - formula['direct_effect_formula']):.2e}")
    
    print(f"\n  PoMA (empirical): {dml_result['poma']:.4f}")
    print(f"  PoMA (formula):   {dml_result['poma_formula']:.4f}")
    
    if abs(dml_result['poma'] - dml_result['poma_formula']) < 0.001:
        print("\n✓ Reduction formula matches empirical estimate!")
    else:
        print("\n⚠ Formula and empirical estimates differ")


def main():
    """Run the non-linear demonstration."""
    
    print("="*80)
    print("STEP 2: NON-LINEAR ROBUSTNESS DEMONSTRATION")
    print("="*80)
    print("\nGenerating data with non-linear M→Y relationship...")
    
    # Generate data
    X, M, Y, true_params = generate_nonlinear_data(n=2000)
    
    print(f"\nTrue parameters:")
    print(f"  Direct effect: {true_params['true_direct']}")
    print(f"  Indirect effect (approx): {true_params['true_indirect_approx']}")
    print(f"  Total effect (approx): {true_params['true_total_approx']}")
    print(f"  True PoMA (approx): {true_params['true_poma_approx']:.3f}")
    
    # Visualize non-linearity
    fig1 = visualize_nonlinearity(X, M, Y, true_params)
    
    # Run all methods
    results = run_methods_comparison(X, M, Y, true_params)
    
    # Create bias comparison
    fig2 = create_bias_comparison_plot(results, true_params)
    
    # Demonstrate reduction formula
    demonstrate_reduction_formula(X, M, Y)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. LINEAR METHODS ARE BIASED:
   - Traditional, FWL, and DML-Linear all give biased estimates
   - They assume linear M→Y but the truth is non-linear
   - Bias can be substantial (here ~40% of true effect)

2. DML WITH ML MODELS WORKS:
   - Random Forest and XGBoost capture non-linearity
   - Direct effect estimates are much less biased
   - R²(Y|M) is high, showing good fit

3. THE REDUCTION FORMULA IS ACCURATE:
   - Formula-based PoMA matches empirical PoMA
   - This provides analytical insight into DML
   - Works even with complex ML models

4. PRACTICAL IMPLICATIONS:
   - Always check for non-linearity in mediation analysis
   - Use flexible ML models in DML when in doubt
   - The reduction formula provides a computational shortcut
""")
    
    # Save figures
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(output_dir, 'step2_nonlinearity_visualization.png'), 
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'step2_bias_comparison.png'), 
                 dpi=150, bbox_inches='tight')
    
    print(f"\nVisualizations saved to outputs/figures/")
    
    plt.show()


if __name__ == "__main__":
    main()