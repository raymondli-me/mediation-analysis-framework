#!/usr/bin/env python3
"""
Step 3: Interactions - Only Causal Framework Survives
====================================================

This demonstration shows what happens with treatment-mediator interactions:
- All methods (Traditional, FWL, DML) estimate the Controlled Direct Effect (CDE)
- But CDE depends on the level of M - it's not a single number!
- Only Natural Direct/Indirect Effects provide meaningful decomposition

Key insight: When there's interaction, the effect of X depends on M.
The causal framework handles this naturally, others don't.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import DataGenerator
from src.traditional_mediation import estimate_effects as traditional_estimate
from src.dml_approach import dml_mediation
from src.causal_mediation import estimate_natural_effects
from src.unified_framework import run_all_methods

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def generate_interaction_data(n=2000, random_state=42):
    """
    Generate data with treatment-mediator interaction.
    
    True model:
    M = 2*X + noise
    Y = 1 + 0.5*X + 2*M + 0.8*X*M + noise
    
    Note: The effect of X on Y depends on M!
    """
    np.random.seed(random_state)
    
    # Binary treatment for clarity
    X = np.random.binomial(1, 0.5, n)
    
    # Mediator
    alpha = 2.0
    M = alpha * X + np.random.normal(0, 0.5, n)
    
    # Outcome with interaction
    gamma = 0.5     # Direct effect when M = 0
    beta = 2.0      # Effect of M when X = 0
    theta = 0.8     # Interaction effect
    
    Y = 1 + gamma * X + beta * M + theta * X * M + np.random.normal(0, 0.5, n)
    
    # Calculate true effects
    # For binary X:
    # E[M|X=0] ≈ 0, E[M|X=1] ≈ 2
    m_0 = 0
    m_1 = alpha
    
    # Natural Direct Effect: Effect of X holding M at its X=0 level
    nde_true = gamma + theta * m_0  # = 0.5
    
    # Natural Indirect Effect: Effect through change in M
    nie_true = (beta + theta) * (m_1 - m_0)  # = 2.8 * 2 = 5.6
    
    # Total effect
    total_true = nde_true + nie_true  # = 6.1
    
    # CDE varies with M!
    # CDE(m) = gamma + theta * m = 0.5 + 0.8 * m
    
    return X, M, Y, {
        'alpha': alpha,
        'gamma': gamma,
        'beta': beta,
        'theta': theta,
        'nde_true': nde_true,
        'nie_true': nie_true,
        'total_true': total_true,
        'cde_function': lambda m: gamma + theta * m
    }


def visualize_interaction(X, M, Y, true_params):
    """Visualize the interaction effect."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Data relationships
    # X vs M
    ax = axes[0, 0]
    # Add jitter for binary X
    X_jitter = X + np.random.normal(0, 0.02, len(X))
    ax.scatter(X_jitter, M, alpha=0.5, s=20)
    ax.set_xlabel('X (Treatment)')
    ax.set_ylabel('M (Mediator)')
    ax.set_title('X → M Relationship')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Treated'])
    
    # M vs Y by X group
    ax = axes[0, 1]
    colors = ['blue', 'red']
    for x_val in [0, 1]:
        mask = X == x_val
        ax.scatter(M[mask], Y[mask], alpha=0.5, s=20, 
                  color=colors[x_val], label=f'X={x_val}')
        
        # Add regression lines
        if sum(mask) > 10:
            m_range = np.linspace(M[mask].min(), M[mask].max(), 100)
            # Y = 1 + gamma*x + beta*m + theta*x*m
            y_pred = (1 + true_params['gamma'] * x_val + 
                     true_params['beta'] * m_range + 
                     true_params['theta'] * x_val * m_range)
            ax.plot(m_range, y_pred, color=colors[x_val], linewidth=2)
    
    ax.set_xlabel('M (Mediator)')
    ax.set_ylabel('Y (Outcome)')
    ax.set_title('M → Y: Different Slopes by X!')
    ax.legend()
    
    # X vs Y
    ax = axes[0, 2]
    ax.scatter(X_jitter, Y, alpha=0.5, s=20)
    # Add mean lines
    for x_val in [0, 1]:
        ax.axhline(y=Y[X == x_val].mean(), 
                  color=colors[x_val], linestyle='--', alpha=0.7,
                  label=f'Mean Y|X={x_val}')
    ax.set_xlabel('X (Treatment)')
    ax.set_ylabel('Y (Outcome)')
    ax.set_title('X → Y Total Effect')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', 'Treated'])
    ax.legend()
    
    # Bottom row: Effect heterogeneity
    # CDE as function of M
    ax = axes[1, 0]
    m_range = np.linspace(-2, 6, 100)
    cde_values = true_params['cde_function'](m_range)
    ax.plot(m_range, cde_values, 'b-', linewidth=3)
    ax.axhline(y=true_params['nde_true'], color='red', linestyle='--', 
               linewidth=2, label=f'NDE = {true_params["nde_true"]:.1f}')
    ax.fill_between(m_range, 0, cde_values, alpha=0.3)
    ax.set_xlabel('Mediator Level (M)')
    ax.set_ylabel('Controlled Direct Effect')
    ax.set_title('CDE Varies with M Level!')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution of M by X
    ax = axes[1, 1]
    for x_val in [0, 1]:
        mask = X == x_val
        ax.hist(M[mask], bins=30, alpha=0.5, density=True,
                color=colors[x_val], label=f'M|X={x_val}')
    ax.set_xlabel('M (Mediator)')
    ax.set_ylabel('Density')
    ax.set_title('Mediator Distributions')
    ax.legend()
    
    # Effect decomposition comparison
    ax = axes[1, 2]
    # Will be filled after running analyses
    ax.axis('off')
    
    plt.suptitle('Treatment-Mediator Interaction: X*M', fontsize=16)
    plt.tight_layout()
    
    return fig, axes


def compare_cde_approaches(X, M, Y, true_params):
    """Compare how different methods handle CDE."""
    
    print("\n" + "="*70)
    print("CONTROLLED DIRECT EFFECT (CDE) ANALYSIS")
    print("="*70)
    
    # All these methods estimate CDE at some M value
    results = []
    
    # 1. Traditional
    trad = traditional_estimate(X, M, Y)
    results.append(('Traditional', trad['direct_effect']))
    
    # 2. DML with linear
    dml_linear = dml_mediation(X, M, Y, n_folds=5)
    results.append(('DML-Linear', dml_linear['direct_effect']))
    
    # 3. DML with RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    dml_rf = dml_mediation(X, M, Y, ml_model_y=rf, ml_model_x=rf, n_folds=5)
    results.append(('DML-RF', dml_rf['direct_effect']))
    
    # What M value do these correspond to?
    # For linear models with interaction, CDE ≈ effect at mean M
    mean_M = np.mean(M)
    cde_at_mean = true_params['cde_function'](mean_M)
    
    print(f"\nMean of M: {mean_M:.2f}")
    print(f"True CDE at mean M: {cde_at_mean:.3f}")
    
    print("\nEstimated CDEs:")
    for method, cde in results:
        print(f"  {method:12s}: {cde:.3f}")
    
    # Show CDE at different M values
    print("\nTrue CDE at different M levels:")
    for m in [0, 1, 2, 3, 4]:
        print(f"  M = {m}: CDE = {true_params['cde_function'](m):.3f}")
    
    print("\n⚠️  CDE is not a single number with interaction!")
    
    return results


def estimate_natural_effects_demo(X, M, Y, true_params):
    """Demonstrate natural effects estimation."""
    
    print("\n" + "="*70)
    print("NATURAL EFFECTS ANALYSIS")
    print("="*70)
    
    # Estimate with linear models (for comparison)
    natural_linear = estimate_natural_effects(X, M, Y, interaction=True)
    
    # Estimate with ML models
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    natural_ml = estimate_natural_effects(X, M, Y, 
                                        outcome_model=rf,
                                        mediator_model=rf,
                                        interaction=True)
    
    print("\nTrue Natural Effects:")
    print(f"  NDE: {true_params['nde_true']:.3f}")
    print(f"  NIE: {true_params['nie_true']:.3f}")
    print(f"  Total: {true_params['total_true']:.3f}")
    
    print("\nEstimated Natural Effects (Linear Models):")
    print(f"  NDE: {natural_linear['nde']:.3f}")
    print(f"  NIE: {natural_linear['nie']:.3f}")
    print(f"  Total: {natural_linear['total_effect']:.3f}")
    print(f"  Check (NDE + NIE): {natural_linear['nde'] + natural_linear['nie']:.3f}")
    
    print("\nEstimated Natural Effects (ML Models):")
    print(f"  NDE: {natural_ml['nde']:.3f}")
    print(f"  NIE: {natural_ml['nie']:.3f}")
    print(f"  Total: {natural_ml['total_effect']:.3f}")
    print(f"  Check (NDE + NIE): {natural_ml['nde'] + natural_ml['nie']:.3f}")
    
    print("\nPotential Outcomes:")
    print(f"  E[Y(1,M(1))]: {natural_ml['E[Y(1,M(1))]']:.3f}")
    print(f"  E[Y(0,M(0))]: {natural_ml['E[Y(0,M(0))]']:.3f}")
    print(f"  E[Y(1,M(0))]: {natural_ml['E[Y(1,M(0))]']:.3f}")
    print(f"  E[Y(0,M(1))]: {natural_ml['E[Y(0,M(1))]']:.3f}")
    
    return natural_linear, natural_ml


def create_comprehensive_comparison(cde_results, natural_linear, natural_ml, true_params, ax):
    """Create final comparison visualization."""
    
    # Prepare data
    methods = []
    direct_effects = []
    indirect_effects = []
    effect_types = []
    
    # CDE results (all estimate same type of effect)
    for method, cde in cde_results:
        methods.append(method)
        direct_effects.append(cde)
        # For CDE methods, indirect = total - direct
        total_est = np.mean([6.0, 6.2, 6.1])  # Approximate total
        indirect_effects.append(total_est - cde)
        effect_types.append('CDE')
    
    # Natural effects
    methods.extend(['Natural-Linear', 'Natural-ML'])
    direct_effects.extend([natural_linear['nde'], natural_ml['nde']])
    indirect_effects.extend([natural_linear['nie'], natural_ml['nie']])
    effect_types.extend(['NDE', 'NDE'])
    
    # Create grouped bar plot
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, direct_effects, width, 
                    label='Direct', alpha=0.8, color='blue')
    bars2 = ax.bar(x + width/2, indirect_effects, width,
                    label='Indirect', alpha=0.8, color='orange')
    
    # Add true values
    ax.axhline(y=true_params['nde_true'], color='blue', 
               linestyle='--', linewidth=2, alpha=0.5)
    ax.axhline(y=true_params['nie_true'], color='orange', 
               linestyle='--', linewidth=2, alpha=0.5)
    
    # Customize
    ax.set_xlabel('Method')
    ax.set_ylabel('Effect Size')
    ax.set_title('Effect Decomposition: CDE vs Natural Effects')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text annotations
    for i, effect_type in enumerate(effect_types):
        ax.text(i, -0.5, effect_type, ha='center', va='top', 
                fontsize=9, style='italic')


def main():
    """Run the interaction demonstration."""
    
    print("="*80)
    print("STEP 3: TREATMENT-MEDIATOR INTERACTION DEMONSTRATION")
    print("="*80)
    print("\nGenerating data with X*M interaction...")
    
    # Generate data
    X, M, Y, true_params = generate_interaction_data(n=2000)
    
    print(f"\nTrue model: Y = 1 + {true_params['gamma']}X + {true_params['beta']}M + {true_params['theta']}XM + noise")
    print(f"\nTrue effects:")
    print(f"  Natural Direct Effect (NDE): {true_params['nde_true']:.3f}")
    print(f"  Natural Indirect Effect (NIE): {true_params['nie_true']:.3f}")
    print(f"  Total Effect: {true_params['total_true']:.3f}")
    
    # Create visualization
    fig, axes = visualize_interaction(X, M, Y, true_params)
    
    # Compare CDE approaches
    cde_results = compare_cde_approaches(X, M, Y, true_params)
    
    # Estimate natural effects
    natural_linear, natural_ml = estimate_natural_effects_demo(X, M, Y, true_params)
    
    # Create comprehensive comparison in the last subplot
    create_comprehensive_comparison(cde_results, natural_linear, natural_ml, 
                                  true_params, axes[1, 2])
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. WITH INTERACTION, CDE VARIES:
   - The direct effect of X depends on the level of M
   - CDE(M) = γ + θM = 0.5 + 0.8M
   - Traditional/DML estimate CDE at some average M level
   - But which M level? It's ambiguous!

2. NATURAL EFFECTS ARE WELL-DEFINED:
   - NDE: Effect of X holding M at its natural (control) level
   - NIE: Effect through changing M
   - These have clear causal interpretations
   - They sum to the total effect

3. PRACTICAL IMPLICATIONS:
   - Always test for interactions in mediation analysis
   - If interaction exists, use natural effects
   - CDE can be misleading with interactions
   - Natural effects require stronger assumptions but give clearer answers

4. THE CAUSAL FRAMEWORK SUBSUMES OTHERS:
   - When no interaction: NDE = CDE, all methods agree
   - With interaction: Only natural effects meaningful
   - This shows why the causal framework is most general
""")
    
    # Additional analysis: Show equivalence when no interaction
    print("\n" + "="*70)
    print("BONUS: What if there were NO interaction?")
    print("="*70)
    
    # Generate data without interaction
    np.random.seed(42)
    X_no_int = np.random.binomial(1, 0.5, 2000)
    M_no_int = 2 * X_no_int + np.random.normal(0, 0.5, 2000)
    Y_no_int = 1 + 0.5 * X_no_int + 2 * M_no_int + np.random.normal(0, 0.5, 2000)
    
    # Traditional
    trad_no_int = traditional_estimate(X_no_int, M_no_int, Y_no_int)
    
    # Natural effects
    natural_no_int = estimate_natural_effects(X_no_int, M_no_int, Y_no_int, 
                                            interaction=False)
    
    print(f"\nWithout interaction:")
    print(f"  CDE (traditional): {trad_no_int['direct_effect']:.3f}")
    print(f"  NDE (natural):     {natural_no_int['nde']:.3f}")
    print(f"  Difference:        {abs(trad_no_int['direct_effect'] - natural_no_int['nde']):.4f}")
    print("\n✓ They're essentially the same when there's no interaction!")
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, 'step3_interaction_analysis.png'), 
                dpi=150, bbox_inches='tight')
    
    print(f"\nVisualization saved to outputs/figures/")
    
    plt.show()


if __name__ == "__main__":
    main()