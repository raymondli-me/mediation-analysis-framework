#!/usr/bin/env python3
"""
Step 1: Establish Equivalence in a Linear World
===============================================

This demonstration shows that in a simple linear world with no treatment-mediator
interaction, the following three estimation methods are equivalent:

1. Traditional Mediation (Baron & Kenny)
2. Frisch-Waugh-Lovell (FWL) 
3. Double Machine Learning (DML) with linear models

They all estimate the same "controlled direct effect" (CDE).
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import DataGenerator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns


def traditional_mediation(X, M, Y):
    """
    Traditional Baron & Kenny approach:
    1. Total effect: Y ~ X
    2. Direct effect: Y ~ X + M
    """
    # Total effect
    lr_total = LinearRegression()
    lr_total.fit(X.reshape(-1, 1), Y)
    beta_total = lr_total.coef_[0]
    
    # Direct effect (controlling for M)
    XM = np.column_stack([X, M])
    lr_direct = LinearRegression()
    lr_direct.fit(XM, Y)
    beta_direct = lr_direct.coef_[0]  # Coefficient on X
    beta_m = lr_direct.coef_[1]       # Coefficient on M
    
    # Indirect effect
    beta_indirect = beta_total - beta_direct
    
    # PoMA
    poma = beta_indirect / beta_total if abs(beta_total) > 1e-10 else np.nan
    
    return {
        'method': 'Traditional',
        'beta_total': beta_total,
        'beta_direct': beta_direct,
        'beta_indirect': beta_indirect,
        'beta_m': beta_m,
        'poma': poma
    }


def fwl_approach(X, M, Y):
    """
    Frisch-Waugh-Lovell approach:
    1. Residualize Y on M: e_Y = Y - E[Y|M]
    2. Residualize X on M: e_X = X - E[X|M]
    3. Regress e_Y on e_X
    """
    # Total effect (same as traditional)
    lr_total = LinearRegression()
    lr_total.fit(X.reshape(-1, 1), Y)
    beta_total = lr_total.coef_[0]
    
    # FWL for direct effect
    # Step 1: Y residuals after removing M
    lr_y_m = LinearRegression()
    lr_y_m.fit(M.reshape(-1, 1), Y)
    e_Y = Y - lr_y_m.predict(M.reshape(-1, 1))
    
    # Step 2: X residuals after removing M
    lr_x_m = LinearRegression()
    lr_x_m.fit(M.reshape(-1, 1), X)
    e_X = X - lr_x_m.predict(M.reshape(-1, 1))
    
    # Step 3: Residual regression
    lr_residual = LinearRegression()
    lr_residual.fit(e_X.reshape(-1, 1), e_Y)
    beta_direct = lr_residual.coef_[0]
    
    # Indirect effect and PoMA
    beta_indirect = beta_total - beta_direct
    poma = beta_indirect / beta_total if abs(beta_total) > 1e-10 else np.nan
    
    return {
        'method': 'FWL',
        'beta_total': beta_total,
        'beta_direct': beta_direct,
        'beta_indirect': beta_indirect,
        'poma': poma,
        'residual_var_Y': np.var(e_Y),
        'residual_var_X': np.var(e_X)
    }


def dml_approach(X, M, Y, n_folds=5):
    """
    Double Machine Learning approach:
    Cross-fitted version of FWL using linear models
    """
    n = len(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Total effect (same as traditional)
    lr_total = LinearRegression()
    lr_total.fit(X.reshape(-1, 1), Y)
    beta_total = lr_total.coef_[0]
    
    # Cross-fitted residuals
    Y_hat = np.zeros(n)
    X_hat = np.zeros(n)
    
    for train_idx, test_idx in kf.split(X):
        # Train models on fold
        lr_y_m = LinearRegression()
        lr_y_m.fit(M[train_idx].reshape(-1, 1), Y[train_idx])
        
        lr_x_m = LinearRegression()
        lr_x_m.fit(M[train_idx].reshape(-1, 1), X[train_idx])
        
        # Predict on held-out fold
        Y_hat[test_idx] = lr_y_m.predict(M[test_idx].reshape(-1, 1))
        X_hat[test_idx] = lr_x_m.predict(M[test_idx].reshape(-1, 1))
    
    # Calculate residuals
    e_Y = Y - Y_hat
    e_X = X - X_hat
    
    # Final regression on residuals
    lr_residual = LinearRegression()
    lr_residual.fit(e_X.reshape(-1, 1), e_Y)
    beta_direct = lr_residual.coef_[0]
    
    # Also calculate using covariance formula
    beta_direct_cov = np.cov(e_Y, e_X)[0, 1] / np.var(e_X) if np.var(e_X) > 1e-10 else 0
    
    # Indirect effect and PoMA
    beta_indirect = beta_total - beta_direct
    poma = beta_indirect / beta_total if abs(beta_total) > 1e-10 else np.nan
    
    return {
        'method': 'DML',
        'beta_total': beta_total,
        'beta_direct': beta_direct,
        'beta_direct_cov': beta_direct_cov,
        'beta_indirect': beta_indirect,
        'poma': poma,
        'n_folds': n_folds
    }


def visualize_equivalence(results, data):
    """Create visualization showing equivalence"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Data relationships
    X, M, Y = data.X, data.M, data.Y
    
    # X vs M
    ax = axes[0, 0]
    ax.scatter(X, M, alpha=0.5, s=20)
    ax.plot(X, data.true_effects['alpha'] * X, 'r-', linewidth=2, label='True: M = 1.5X')
    ax.set_xlabel('X')
    ax.set_ylabel('M')
    ax.set_title('X → M Relationship')
    ax.legend()
    
    # M vs Y
    ax = axes[0, 1]
    ax.scatter(M, Y, alpha=0.5, s=20)
    # Partial regression line (holding X constant)
    M_sorted = np.sort(M)
    Y_partial = data.true_effects['beta'] * M_sorted + data.true_effects['gamma'] * np.mean(X)
    ax.plot(M_sorted, Y_partial, 'r-', linewidth=2, label='True: Y = 0.5X + 2M')
    ax.set_xlabel('M')
    ax.set_ylabel('Y')
    ax.set_title('M → Y Relationship')
    ax.legend()
    
    # X vs Y
    ax = axes[0, 2]
    ax.scatter(X, Y, alpha=0.5, s=20)
    ax.plot(X, data.true_effects['total'] * X, 'r-', linewidth=2, 
            label=f"True total: {data.true_effects['total']:.2f}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('X → Y Total Relationship')
    ax.legend()
    
    # Bottom row: Results comparison
    # Extract values
    methods = [r['method'] for r in results]
    beta_totals = [r['beta_total'] for r in results]
    beta_directs = [r['beta_direct'] for r in results]
    beta_indirects = [r['beta_indirect'] for r in results]
    pomas = [r['poma'] for r in results]
    
    # Beta comparisons
    ax = axes[1, 0]
    x_pos = np.arange(len(methods))
    width = 0.25
    
    ax.bar(x_pos - width, beta_totals, width, label='Total', alpha=0.8)
    ax.bar(x_pos, beta_directs, width, label='Direct', alpha=0.8)
    ax.bar(x_pos + width, beta_indirects, width, label='Indirect', alpha=0.8)
    
    # Add true values as horizontal lines
    ax.axhline(y=data.true_effects['total'], color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=data.true_effects['direct'], color='black', linestyle='--', alpha=0.5)
    ax.axhline(y=data.true_effects['indirect'], color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Effect Size')
    ax.set_title('Effect Estimates')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # PoMA comparison
    ax = axes[1, 1]
    ax.bar(methods, pomas, alpha=0.8, color='green')
    ax.axhline(y=data.true_effects['poma'], color='red', linestyle='--', 
               label=f"True: {data.true_effects['poma']:.3f}")
    ax.set_ylabel('PoMA')
    ax.set_title('Proportion Mediated (PoMA)')
    ax.legend()
    
    # Numerical differences
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate differences
    diff_text = "Numerical Differences (from Traditional):\n\n"
    trad_direct = results[0]['beta_direct']
    
    for i, result in enumerate(results[1:], 1):
        diff = abs(result['beta_direct'] - trad_direct)
        diff_text += f"{result['method']} vs Traditional: {diff:.2e}\n"
    
    diff_text += f"\nAll differences < 1e-10: {'✓' if all(abs(r['beta_direct'] - trad_direct) < 1e-10 for r in results[1:]) else '✗'}"
    
    ax.text(0.1, 0.5, diff_text, fontsize=12, family='monospace',
            verticalalignment='center')
    ax.set_title('Equivalence Check')
    
    plt.suptitle('Linear World: Perfect Equivalence of Three Approaches', fontsize=16)
    plt.tight_layout()
    
    return fig


def main():
    """Run the linear equivalence demonstration"""
    
    print("="*80)
    print("STEP 1: LINEAR EQUIVALENCE DEMONSTRATION")
    print("="*80)
    print("\nGenerating linear mediation data...")
    
    # Generate data
    gen = DataGenerator(random_state=42)
    data = gen.linear_mediation(n=2000, alpha=1.5, beta=2.0, gamma=0.5)
    
    print(f"\nTrue model parameters:")
    print(f"  X → M: α = {data.true_effects['alpha']}")
    print(f"  M → Y: β = {data.true_effects['beta']}")
    print(f"  X → Y: γ = {data.true_effects['gamma']} (direct)")
    print(f"\nTrue effects:")
    print(f"  Total effect: {data.true_effects['total']:.3f}")
    print(f"  Direct effect: {data.true_effects['direct']:.3f}")
    print(f"  Indirect effect: {data.true_effects['indirect']:.3f}")
    print(f"  True PoMA: {data.true_effects['poma']:.3f}")
    
    # Run all three approaches
    print("\n" + "-"*60)
    print("Running three estimation approaches...")
    
    results = []
    
    # 1. Traditional
    print("\n1. Traditional Mediation Analysis...")
    trad_result = traditional_mediation(data.X, data.M, data.Y)
    results.append(trad_result)
    print(f"   β_direct = {trad_result['beta_direct']:.6f}")
    print(f"   PoMA = {trad_result['poma']:.6f}")
    
    # 2. FWL
    print("\n2. Frisch-Waugh-Lovell Approach...")
    fwl_result = fwl_approach(data.X, data.M, data.Y)
    results.append(fwl_result)
    print(f"   β_direct = {fwl_result['beta_direct']:.6f}")
    print(f"   PoMA = {fwl_result['poma']:.6f}")
    
    # 3. DML
    print("\n3. Double Machine Learning (with linear models)...")
    dml_result = dml_approach(data.X, data.M, data.Y)
    results.append(dml_result)
    print(f"   β_direct = {dml_result['beta_direct']:.6f}")
    print(f"   β_direct (cov formula) = {dml_result['beta_direct_cov']:.6f}")
    print(f"   PoMA = {dml_result['poma']:.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    
    print("\nDirect Effect Estimates:")
    for result in results:
        print(f"  {result['method']:12s}: {result['beta_direct']:.10f}")
    
    print(f"\nTrue direct effect: {data.true_effects['direct']:.10f}")
    
    # Check equivalence
    direct_effects = [r['beta_direct'] for r in results]
    max_diff = max(direct_effects) - min(direct_effects)
    print(f"\nMaximum difference: {max_diff:.2e}")
    print(f"Numerically equivalent? {'YES' if max_diff < 1e-10 else 'NO'}")
    
    # Theoretical explanation
    print("\n" + "="*60)
    print("THEORETICAL EXPLANATION")
    print("="*60)
    print("""
The three approaches are mathematically equivalent in the linear case:

1. Traditional: Estimates β₁ from Y = β₀ + β₁X + β₂M + ε

2. FWL: By the Frisch-Waugh-Lovell theorem:
   - β₁ from Y ~ X + M equals
   - coefficient from regressing (Y - E[Y|M]) on (X - E[X|M])
   
3. DML: Cross-fitting doesn't change the result when using linear models
   - It's just a sample-splitting version of FWL
   - Helps with inference but gives same point estimate

All three estimate the "Controlled Direct Effect" (CDE):
- The effect of X on Y holding M constant
- In linear models with no interaction, CDE = NDE
""")
    
    # Create visualization
    fig = visualize_equivalence(results, data)
    
    # Save outputs
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'step1_linear_equivalence.png'), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to outputs/figures/step1_linear_equivalence.png")
    
    plt.show()


if __name__ == "__main__":
    main()