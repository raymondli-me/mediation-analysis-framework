#!/usr/bin/env python3
"""
Edge Cases in Mediation Analysis
================================

This demonstration explores challenging scenarios where mediation analysis
can produce counterintuitive or unstable results:

1. Symmetric relationships (X² → M → sin(M))
2. Suppression effects (sign reversals)
3. Near-zero effects (numerical instability)
4. Complete mediation vs no mediation
5. Inconsistent mediation

These cases highlight the importance of:
- Understanding your data before analysis
- Using appropriate methods for each scenario
- Being cautious with PoMA interpretation
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traditional_mediation import estimate_effects as traditional_estimate
from src.dml_approach import dml_mediation
from src.causal_mediation import estimate_natural_effects

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def generate_symmetric_data(n=1000, random_state=42):
    """
    Generate data with symmetric non-linear relationship.
    X² → M → sin(M) creates near-zero correlation between X and Y.
    """
    np.random.seed(random_state)
    
    # Symmetric X distribution
    X = np.random.uniform(-2, 2, n)
    
    # X² → M (symmetric transformation)
    M = X**2 + np.random.normal(0, 0.2, n)
    
    # sin(M) → Y (periodic transformation)
    Y = np.sin(M) + np.random.normal(0, 0.1, n)
    
    # Calculate correlations
    corr_XY = np.corrcoef(X, Y)[0, 1]
    corr_XM = np.corrcoef(X, M)[0, 1]
    corr_MY = np.corrcoef(M, Y)[0, 1]
    
    print(f"\nSymmetric relationship correlations:")
    print(f"  cor(X,Y) = {corr_XY:.4f} (near zero!)")
    print(f"  cor(X,M) = {corr_XM:.4f} (near zero!)")
    print(f"  cor(M,Y) = {corr_MY:.4f}")
    
    return X, M, Y


def generate_suppression_data(n=1000, random_state=42):
    """
    Generate data with suppression effect.
    Direct and indirect effects have opposite signs.
    """
    np.random.seed(random_state)
    
    X = np.random.normal(0, 1, n)
    
    # Strong positive X→M
    M = 2 * X + np.random.normal(0, 0.5, n)
    
    # Negative direct, positive indirect
    # Y = -X + 0.8*M + noise
    # Total effect = -1 + 0.8*2 = 0.6
    Y = -1 * X + 0.8 * M + np.random.normal(0, 0.5, n)
    
    true_direct = -1.0
    true_indirect = 0.8 * 2.0  # 1.6
    true_total = true_direct + true_indirect  # 0.6
    
    print(f"\nSuppression effect true values:")
    print(f"  Direct: {true_direct:.2f} (negative)")
    print(f"  Indirect: {true_indirect:.2f} (positive)")
    print(f"  Total: {true_total:.2f}")
    print(f"  PoMA: {true_indirect/true_total:.2f} = 267%!")
    
    return X, M, Y, {'direct': true_direct, 'indirect': true_indirect, 'total': true_total}


def generate_nearzero_direct(n=1000, random_state=42):
    """
    Generate data with near-zero direct effect.
    Almost complete mediation.
    """
    np.random.seed(random_state)
    
    X = np.random.normal(0, 1, n)
    M = 1.5 * X + np.random.normal(0, 0.5, n)
    
    # Tiny direct effect
    Y = 0.01 * X + 0.8 * M + np.random.normal(0, 0.5, n)
    
    true_direct = 0.01
    true_indirect = 0.8 * 1.5  # 1.2
    true_total = true_direct + true_indirect  # 1.21
    true_poma = true_indirect / true_total  # 0.992
    
    print(f"\nNear-zero direct effect:")
    print(f"  Direct: {true_direct:.3f}")
    print(f"  Indirect: {true_indirect:.3f}")
    print(f"  Total: {true_total:.3f}")
    print(f"  PoMA: {true_poma:.3f} (99.2% mediated)")
    
    return X, M, Y, {'direct': true_direct, 'indirect': true_indirect, 
                     'total': true_total, 'poma': true_poma}


def generate_nearzero_indirect(n=1000, random_state=42):
    """
    Generate data with near-zero indirect effect.
    Almost no mediation.
    """
    np.random.seed(random_state)
    
    X = np.random.normal(0, 1, n)
    
    # Weak X→M relationship
    M = 0.1 * X + np.random.normal(0, 1, n)
    
    # Strong direct, weak indirect through M
    Y = 1.5 * X + 0.1 * M + np.random.normal(0, 0.5, n)
    
    true_direct = 1.5
    true_indirect = 0.1 * 0.1  # 0.01
    true_total = true_direct + true_indirect  # 1.51
    true_poma = true_indirect / true_total  # 0.0066
    
    print(f"\nNear-zero indirect effect:")
    print(f"  Direct: {true_direct:.3f}")
    print(f"  Indirect: {true_indirect:.3f}")
    print(f"  Total: {true_total:.3f}")
    print(f"  PoMA: {true_poma:.4f} (0.66% mediated)")
    
    return X, M, Y, {'direct': true_direct, 'indirect': true_indirect,
                     'total': true_total, 'poma': true_poma}


def generate_inconsistent_mediation(n=1000, random_state=42):
    """
    Generate data with inconsistent mediation.
    X affects M, M affects Y, but X doesn't affect Y directly.
    This violates the assumptions of mediation.
    """
    np.random.seed(random_state)
    
    # Confounded relationship
    U = np.random.normal(0, 1, n)  # Unobserved confounder
    
    X = U + np.random.normal(0, 0.5, n)
    M = 1.5 * X + np.random.normal(0, 0.5, n)
    
    # Y depends on U and M, but not X directly
    Y = 0 * X + 0.8 * M - 0.5 * U + np.random.normal(0, 0.5, n)
    
    print(f"\nInconsistent mediation (confounded):")
    print(f"  True model: Y = 0*X + 0.8*M - 0.5*U")
    print(f"  But we don't observe U!")
    
    return X, M, Y


def visualize_edge_case(X, M, Y, title, ax_row):
    """Visualize an edge case scenario."""
    # Scatter plots
    ax1, ax2, ax3 = ax_row
    
    # X vs M
    ax1.scatter(X, M, alpha=0.3, s=10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('M')
    ax1.set_title(f'{title}\nX → M')
    
    # Add correlation
    corr_XM = np.corrcoef(X, M)[0, 1]
    ax1.text(0.05, 0.95, f'r = {corr_XM:.3f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # M vs Y
    ax2.scatter(M, Y, alpha=0.3, s=10, color='orange')
    ax2.set_xlabel('M')
    ax2.set_ylabel('Y')
    ax2.set_title('M → Y')
    
    corr_MY = np.corrcoef(M, Y)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr_MY:.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # X vs Y
    ax3.scatter(X, Y, alpha=0.3, s=10, color='green')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('X → Y (Total)')
    
    corr_XY = np.corrcoef(X, Y)[0, 1]
    ax3.text(0.05, 0.95, f'r = {corr_XY:.3f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add regression lines
    from sklearn.linear_model import LinearRegression
    for ax, x_data, y_data in [(ax1, X, M), (ax2, M, Y), (ax3, X, Y)]:
        lr = LinearRegression()
        lr.fit(x_data.reshape(-1, 1), y_data)
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_pred = lr.predict(x_range.reshape(-1, 1))
        ax.plot(x_range, y_pred, 'r--', linewidth=2, alpha=0.7)
        ax.grid(True, alpha=0.3)


def analyze_edge_case(X, M, Y, case_name, true_effects=None):
    """Analyze an edge case with multiple methods."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {case_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Traditional
    try:
        trad = traditional_estimate(X, M, Y)
        results['Traditional'] = {
            'direct': trad['direct_effect'],
            'indirect': trad['indirect_effect'],
            'total': trad['total_effect'],
            'poma': trad['poma']
        }
    except Exception as e:
        results['Traditional'] = {'error': str(e)}
    
    # 2. DML Linear
    try:
        dml_lin = dml_mediation(X, M, Y, n_folds=5)
        results['DML-Linear'] = {
            'direct': dml_lin['direct_effect'],
            'indirect': dml_lin['indirect_effect'],
            'total': dml_lin['total_effect'],
            'poma': dml_lin['poma'],
            'poma_formula': dml_lin['poma_formula']
        }
    except Exception as e:
        results['DML-Linear'] = {'error': str(e)}
    
    # 3. DML RF
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        dml_rf = dml_mediation(X, M, Y, ml_model_y=rf, ml_model_x=rf, n_folds=5)
        results['DML-RF'] = {
            'direct': dml_rf['direct_effect'],
            'indirect': dml_rf['indirect_effect'],
            'total': dml_rf['total_effect'],
            'poma': dml_rf['poma'],
            'r2_y': dml_rf['r2_y_given_m'],
            'r2_x': dml_rf['r2_x_given_m']
        }
    except Exception as e:
        results['DML-RF'] = {'error': str(e)}
    
    # 4. Natural Effects
    try:
        natural = estimate_natural_effects(X, M, Y, interaction=False)
        results['Natural'] = {
            'direct': natural['nde'],
            'indirect': natural['nie'],
            'total': natural['total_effect'],
            'poma': natural['prop_mediated']
        }
    except Exception as e:
        results['Natural'] = {'error': str(e)}
    
    # Print results
    print("\nEstimated effects:")
    print(f"{'Method':<15} {'Direct':>10} {'Indirect':>10} {'Total':>10} {'PoMA':>10}")
    print("-" * 60)
    
    for method, res in results.items():
        if 'error' in res:
            print(f"{method:<15} {'ERROR':>10}")
        else:
            print(f"{method:<15} {res['direct']:>10.3f} {res['indirect']:>10.3f} "
                  f"{res['total']:>10.3f} {res['poma']:>10.1%}")
    
    if true_effects:
        print("-" * 60)
        print(f"{'TRUE':<15} {true_effects.get('direct', 0):>10.3f} "
              f"{true_effects.get('indirect', 0):>10.3f} "
              f"{true_effects.get('total', 0):>10.3f} "
              f"{true_effects.get('poma', true_effects.get('indirect', 0)/true_effects.get('total', 1)):>10.1%}")
    
    return results


def create_edge_cases_summary():
    """Create comprehensive edge cases analysis."""
    fig = plt.figure(figsize=(18, 20))
    
    # 1. Symmetric relationship
    print("\n" + "="*80)
    print("EDGE CASE 1: SYMMETRIC RELATIONSHIP")
    print("="*80)
    X1, M1, Y1 = generate_symmetric_data()
    ax_row1 = [plt.subplot(6, 3, i) for i in [1, 2, 3]]
    visualize_edge_case(X1, M1, Y1, "Symmetric: X² → M → sin(M)", ax_row1)
    results1 = analyze_edge_case(X1, M1, Y1, "Symmetric Relationship")
    
    # 2. Suppression effect
    print("\n" + "="*80)
    print("EDGE CASE 2: SUPPRESSION EFFECT")
    print("="*80)
    X2, M2, Y2, true2 = generate_suppression_data()
    ax_row2 = [plt.subplot(6, 3, i) for i in [4, 5, 6]]
    visualize_edge_case(X2, M2, Y2, "Suppression: Opposite Signs", ax_row2)
    results2 = analyze_edge_case(X2, M2, Y2, "Suppression Effect", true2)
    
    # 3. Near-zero direct
    print("\n" + "="*80)
    print("EDGE CASE 3: NEAR-ZERO DIRECT EFFECT")
    print("="*80)
    X3, M3, Y3, true3 = generate_nearzero_direct()
    ax_row3 = [plt.subplot(6, 3, i) for i in [7, 8, 9]]
    visualize_edge_case(X3, M3, Y3, "Near-Zero Direct", ax_row3)
    results3 = analyze_edge_case(X3, M3, Y3, "Near-Zero Direct", true3)
    
    # 4. Near-zero indirect
    print("\n" + "="*80)
    print("EDGE CASE 4: NEAR-ZERO INDIRECT EFFECT")
    print("="*80)
    X4, M4, Y4, true4 = generate_nearzero_indirect()
    ax_row4 = [plt.subplot(6, 3, i) for i in [10, 11, 12]]
    visualize_edge_case(X4, M4, Y4, "Near-Zero Indirect", ax_row4)
    results4 = analyze_edge_case(X4, M4, Y4, "Near-Zero Indirect", true4)
    
    # 5. Inconsistent mediation
    print("\n" + "="*80)
    print("EDGE CASE 5: INCONSISTENT MEDIATION")
    print("="*80)
    X5, M5, Y5 = generate_inconsistent_mediation()
    ax_row5 = [plt.subplot(6, 3, i) for i in [13, 14, 15]]
    visualize_edge_case(X5, M5, Y5, "Inconsistent (Confounded)", ax_row5)
    results5 = analyze_edge_case(X5, M5, Y5, "Inconsistent Mediation")
    
    # 6. Summary plot - PoMA stability
    ax_summary = plt.subplot(6, 1, 6)
    
    # Collect PoMA values
    cases = ['Symmetric', 'Suppression', 'Near-0 Direct', 'Near-0 Indirect', 'Inconsistent']
    methods = ['Traditional', 'DML-Linear', 'DML-RF', 'Natural']
    all_results = [results1, results2, results3, results4, results5]
    
    poma_data = []
    for i, (case, results) in enumerate(zip(cases, all_results)):
        for method in methods:
            if method in results and 'poma' in results[method]:
                poma = results[method]['poma']
                if not np.isnan(poma) and abs(poma) < 10:  # Exclude extreme outliers for visualization
                    poma_data.append({
                        'Case': case,
                        'Method': method,
                        'PoMA': poma
                    })
    
    # Create grouped bar plot
    import pandas as pd
    poma_df = pd.DataFrame(poma_data)
    poma_pivot = poma_df.pivot(index='Case', columns='Method', values='PoMA')
    
    poma_pivot.plot(kind='bar', ax=ax_summary)
    ax_summary.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_summary.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax_summary.set_ylabel('PoMA')
    ax_summary.set_title('PoMA Stability Across Edge Cases')
    ax_summary.legend(loc='upper right')
    ax_summary.grid(True, alpha=0.3)
    ax_summary.set_ylim(-2, 3)
    
    plt.suptitle('Edge Cases in Mediation Analysis', fontsize=16)
    plt.tight_layout()
    
    return fig


def main():
    """Run edge cases demonstration."""
    print("="*80)
    print("EDGE CASES IN MEDIATION ANALYSIS")
    print("="*80)
    print("\nThis demonstration explores challenging scenarios:")
    print("1. Symmetric relationships (near-zero correlations)")
    print("2. Suppression effects (opposite sign effects)")
    print("3. Near-zero effects (numerical instability)")
    print("4. Inconsistent mediation (confounding)")
    
    # Create comprehensive analysis
    fig = create_edge_cases_summary()
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FROM EDGE CASES")
    print("="*80)
    print("""
1. SYMMETRIC RELATIONSHIPS:
   - When cor(X,Y) ≈ 0, PoMA becomes extremely unstable
   - Covariance-based formulas can produce nonsensical values
   - Even ML methods struggle without strong signal

2. SUPPRESSION EFFECTS:
   - PoMA > 100% is possible and meaningful
   - Indicates opposing direct and indirect pathways
   - Important to check effect signs, not just magnitudes

3. NEAR-ZERO EFFECTS:
   - Small denominators cause numerical instability
   - Confidence intervals become crucial
   - Consider clinical/practical significance

4. INCONSISTENT MEDIATION:
   - Violations of assumptions lead to biased estimates
   - Always check causal assumptions before analysis
   - Consider unmeasured confounding

5. PRACTICAL RECOMMENDATIONS:
   - Always visualize your data first
   - Be cautious interpreting PoMA in edge cases
   - Report effect sizes, not just proportions
   - Use multiple methods for robustness
   - Consider the causal model carefully
""")
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, 'edge_cases_analysis.png'), 
                dpi=150, bbox_inches='tight')
    
    print(f"\nVisualization saved to outputs/figures/edge_cases_analysis.png")
    
    plt.show()


if __name__ == "__main__":
    main()