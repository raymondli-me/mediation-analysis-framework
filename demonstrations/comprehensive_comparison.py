#!/usr/bin/env python3
"""
Comprehensive Comparison: All Methods on All Scenarios
======================================================

This demonstration runs all four mediation frameworks on all 11 data scenarios
to create a comprehensive comparison matrix. It shows:

1. When methods agree (linear scenarios)
2. When DML outperforms traditional methods (non-linear scenarios)
3. When only causal framework works (interaction scenarios)
4. Edge cases where all methods struggle

The output is a detailed comparison table and visualizations showing
performance across different data characteristics.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generation import DataGenerator
from src.traditional_mediation import estimate_effects as traditional_estimate
from src.fwl_approach import fwl_regression
from src.dml_approach import dml_mediation
from src.causal_mediation import estimate_natural_effects
from src.unified_framework import run_all_methods, diagnostic_summary

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


def run_comprehensive_comparison(n_samples=2000, random_state=42):
    """
    Run all methods on all scenarios and collect results.
    """
    # Initialize data generator
    generator = DataGenerator(random_state=random_state)
    
    # Results storage
    all_results = []
    
    # Define ML models for DML
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    
    # Scenarios to test
    scenarios = [
        'linear_simple',
        'linear_complex', 
        'nonlinear_smooth',
        'nonlinear_complex',
        'interaction_linear',
        'interaction_nonlinear',
        'confounded',
        'instrumental',
        'zero_direct',
        'zero_indirect',
        'suppression'
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario.replace('_', ' ').title()}")
        print(f"{'='*70}")
        
        # Generate data
        data = generator.generate_scenario(scenario, n=n_samples)
        X = data.X
        M = data.M
        Y = data.Y
        true_effects = data.true_effects
        
        # Print true effects
        print(f"\nTrue effects:")
        for key, value in true_effects.items():
            if not key.startswith('_'):
                print(f"  {key}: {value:.3f}")
        
        # Run diagnostic
        diagnostics = diagnostic_summary(X, M, Y)
        print(f"\nDiagnostics:")
        for rec in diagnostics['recommendations']:
            print(f"  {rec}")
        
        # 1. Traditional mediation
        try:
            trad = traditional_estimate(X, M, Y)
            results_trad = {
                'scenario': scenario,
                'method': 'Traditional',
                'direct_effect': trad['direct_effect'],
                'indirect_effect': trad['indirect_effect'],
                'total_effect': trad['total_effect'],
                'poma': trad['poma'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan)
            }
        except Exception as e:
            results_trad = {
                'scenario': scenario,
                'method': 'Traditional',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_trad)
        
        # 2. FWL
        try:
            fwl = fwl_regression(X, M, Y)
            results_fwl = {
                'scenario': scenario,
                'method': 'FWL',
                'direct_effect': fwl['direct_effect'],
                'indirect_effect': fwl['indirect_effect'],
                'total_effect': fwl['total_effect'],
                'poma': fwl['poma'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan)
            }
        except Exception as e:
            results_fwl = {
                'scenario': scenario,
                'method': 'FWL',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_fwl)
        
        # 3. DML Linear
        try:
            dml_linear = dml_mediation(X, M, Y, n_folds=5, random_state=random_state)
            results_dml_linear = {
                'scenario': scenario,
                'method': 'DML-Linear',
                'direct_effect': dml_linear['direct_effect'],
                'indirect_effect': dml_linear['indirect_effect'],
                'total_effect': dml_linear['total_effect'],
                'poma': dml_linear['poma'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan)
            }
        except Exception as e:
            results_dml_linear = {
                'scenario': scenario,
                'method': 'DML-Linear',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_dml_linear)
        
        # 4. DML with Random Forest
        try:
            dml_rf = dml_mediation(X, M, Y, 
                                  ml_model_y=rf_model,
                                  ml_model_x=rf_model,
                                  n_folds=5, 
                                  random_state=random_state)
            results_dml_rf = {
                'scenario': scenario,
                'method': 'DML-RF',
                'direct_effect': dml_rf['direct_effect'],
                'indirect_effect': dml_rf['indirect_effect'],
                'total_effect': dml_rf['total_effect'],
                'poma': dml_rf['poma'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'r2_y_given_m': dml_rf['r2_y_given_m'],
                'r2_x_given_m': dml_rf['r2_x_given_m']
            }
        except Exception as e:
            results_dml_rf = {
                'scenario': scenario,
                'method': 'DML-RF',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_dml_rf)
        
        # 5. Natural Effects (Linear)
        try:
            natural_linear = estimate_natural_effects(X, M, Y, interaction=True)
            results_natural_linear = {
                'scenario': scenario,
                'method': 'Natural-Linear',
                'direct_effect': natural_linear['nde'],
                'indirect_effect': natural_linear['nie'],
                'total_effect': natural_linear['total_effect'],
                'poma': natural_linear['prop_mediated'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan)
            }
        except Exception as e:
            results_natural_linear = {
                'scenario': scenario,
                'method': 'Natural-Linear',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_natural_linear)
        
        # 6. Natural Effects (RF)
        try:
            natural_rf = estimate_natural_effects(X, M, Y, 
                                                outcome_model=rf_model,
                                                mediator_model=rf_model,
                                                interaction=True)
            results_natural_rf = {
                'scenario': scenario,
                'method': 'Natural-RF',
                'direct_effect': natural_rf['nde'],
                'indirect_effect': natural_rf['nie'],
                'total_effect': natural_rf['total_effect'],
                'poma': natural_rf['prop_mediated'],
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan)
            }
        except Exception as e:
            results_natural_rf = {
                'scenario': scenario,
                'method': 'Natural-RF',
                'direct_effect': np.nan,
                'indirect_effect': np.nan,
                'total_effect': np.nan,
                'poma': np.nan,
                'true_direct': true_effects.get('direct', np.nan),
                'true_indirect': true_effects.get('indirect', np.nan),
                'true_total': true_effects.get('total', np.nan),
                'true_poma': true_effects.get('prop_mediated', np.nan),
                'error': str(e)
            }
        all_results.append(results_natural_rf)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate biases
    results_df['bias_direct'] = results_df['direct_effect'] - results_df['true_direct']
    results_df['bias_indirect'] = results_df['indirect_effect'] - results_df['true_indirect']
    results_df['bias_total'] = results_df['total_effect'] - results_df['true_total']
    results_df['bias_poma'] = results_df['poma'] - results_df['true_poma']
    
    # Calculate relative biases (where possible)
    results_df['rel_bias_direct'] = np.where(
        np.abs(results_df['true_direct']) > 0.1,
        results_df['bias_direct'] / results_df['true_direct'],
        np.nan
    )
    results_df['rel_bias_indirect'] = np.where(
        np.abs(results_df['true_indirect']) > 0.1,
        results_df['bias_indirect'] / results_df['true_indirect'],
        np.nan
    )
    
    return results_df


def create_performance_heatmap(results_df):
    """
    Create a heatmap showing method performance across scenarios.
    """
    # Pivot for direct effect bias
    pivot_direct = results_df.pivot(index='scenario', columns='method', values='bias_direct')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Direct effect bias
    ax = axes[0, 0]
    sns.heatmap(pivot_direct, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'Bias'})
    ax.set_title('Direct Effect Bias')
    ax.set_xlabel('Method')
    ax.set_ylabel('Scenario')
    
    # 2. Indirect effect bias
    ax = axes[0, 1]
    pivot_indirect = results_df.pivot(index='scenario', columns='method', values='bias_indirect')
    sns.heatmap(pivot_indirect, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': 'Bias'})
    ax.set_title('Indirect Effect Bias')
    ax.set_xlabel('Method')
    ax.set_ylabel('Scenario')
    
    # 3. PoMA bias
    ax = axes[1, 0]
    pivot_poma = results_df.pivot(index='scenario', columns='method', values='bias_poma')
    sns.heatmap(pivot_poma, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': 'Bias'})
    ax.set_title('PoMA Bias')
    ax.set_xlabel('Method')
    ax.set_ylabel('Scenario')
    
    # 4. Overall performance (RMS bias)
    ax = axes[1, 1]
    # Calculate RMS bias for each method-scenario
    results_df['rms_bias'] = np.sqrt(
        results_df['bias_direct']**2 + 
        results_df['bias_indirect']**2
    )
    pivot_rms = results_df.pivot(index='scenario', columns='method', values='rms_bias')
    sns.heatmap(pivot_rms, annot=True, fmt='.3f', cmap='Reds',
                ax=ax, cbar_kws={'label': 'RMS Bias'})
    ax.set_title('Overall Performance (RMS Bias)')
    ax.set_xlabel('Method')
    ax.set_ylabel('Scenario')
    
    plt.suptitle('Method Performance Across Scenarios', fontsize=16)
    plt.tight_layout()
    
    return fig


def create_scenario_specific_plots(results_df):
    """
    Create detailed plots for specific interesting scenarios.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Define scenarios to highlight
    highlight_scenarios = [
        ('linear_simple', 'All methods should agree'),
        ('nonlinear_smooth', 'DML-RF should outperform linear'),
        ('interaction_linear', 'Natural effects handle interaction'),
        ('zero_direct', 'Direct effect = 0'),
        ('zero_indirect', 'Indirect effect = 0'),
        ('suppression', 'Suppression effect')
    ]
    
    for idx, (scenario, title) in enumerate(highlight_scenarios):
        ax = axes[idx // 3, idx % 3]
        
        # Filter data for this scenario
        scenario_data = results_df[results_df['scenario'] == scenario].copy()
        
        # Create grouped bar plot
        x = np.arange(len(scenario_data))
        width = 0.2
        
        # Direct effects
        bars1 = ax.bar(x - width, scenario_data['direct_effect'], 
                       width, label='Direct', alpha=0.8, color='blue')
        
        # Indirect effects
        bars2 = ax.bar(x, scenario_data['indirect_effect'],
                       width, label='Indirect', alpha=0.8, color='orange')
        
        # Total effects
        bars3 = ax.bar(x + width, scenario_data['total_effect'],
                       width, label='Total', alpha=0.8, color='green')
        
        # Add true values as horizontal lines
        true_direct = scenario_data['true_direct'].iloc[0]
        true_indirect = scenario_data['true_indirect'].iloc[0]
        true_total = scenario_data['true_total'].iloc[0]
        
        ax.axhline(y=true_direct, color='blue', linestyle='--', alpha=0.7)
        ax.axhline(y=true_indirect, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(y=true_total, color='green', linestyle='--', alpha=0.7)
        
        # Customize
        ax.set_xlabel('Method')
        ax.set_ylabel('Effect Size')
        ax.set_title(f'{scenario.replace("_", " ").title()}\n{title}')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_data['method'], rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scenario-Specific Performance', fontsize=16)
    plt.tight_layout()
    
    return fig


def create_summary_table(results_df):
    """
    Create a summary table of key findings.
    """
    # Group by scenario and calculate summary statistics
    summary = []
    
    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        
        # Find best method for each metric
        best_direct = scenario_data.loc[
            scenario_data['bias_direct'].abs().idxmin(), 'method'
        ]
        best_indirect = scenario_data.loc[
            scenario_data['bias_indirect'].abs().idxmin(), 'method'
        ]
        best_poma = scenario_data.loc[
            scenario_data['bias_poma'].abs().idxmin(), 'method'
        ]
        
        # Check if linear methods fail
        linear_methods = ['Traditional', 'FWL', 'DML-Linear']
        linear_biases = scenario_data[
            scenario_data['method'].isin(linear_methods)
        ]['bias_direct'].abs().mean()
        
        ml_methods = ['DML-RF', 'Natural-RF']
        ml_biases = scenario_data[
            scenario_data['method'].isin(ml_methods)
        ]['bias_direct'].abs().mean()
        
        linear_fails = linear_biases > 0.1 and ml_biases < linear_biases * 0.5
        
        summary.append({
            'scenario': scenario,
            'best_direct': best_direct,
            'best_indirect': best_indirect,
            'best_poma': best_poma,
            'linear_fails': linear_fails,
            'has_interaction': 'interaction' in scenario,
            'true_direct': scenario_data['true_direct'].iloc[0],
            'true_indirect': scenario_data['true_indirect'].iloc[0]
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    print("\n1. When do linear methods fail?")
    failures = summary_df[summary_df['linear_fails']]
    if len(failures) > 0:
        print(f"   Linear methods fail in {len(failures)} scenarios:")
        for _, row in failures.iterrows():
            print(f"   - {row['scenario'].replace('_', ' ').title()}")
    else:
        print("   Linear methods perform adequately in all scenarios")
    
    print("\n2. Best methods by scenario type:")
    print("   Linear scenarios: Traditional/FWL/DML-Linear (all equivalent)")
    print("   Non-linear scenarios: DML-RF or Natural-RF")
    print("   Interaction scenarios: Natural Effects framework")
    
    print("\n3. Edge cases:")
    for _, row in summary_df.iterrows():
        if row['true_direct'] == 0:
            print(f"   - {row['scenario']}: Zero direct effect")
        elif row['true_indirect'] == 0:
            print(f"   - {row['scenario']}: Zero indirect effect")
    
    return summary_df


def main():
    """
    Run comprehensive comparison and create all outputs.
    """
    print("="*80)
    print("COMPREHENSIVE MEDIATION FRAMEWORK COMPARISON")
    print("="*80)
    print("\nThis analysis compares 6 methods across 11 scenarios:")
    print("Methods: Traditional, FWL, DML-Linear, DML-RF, Natural-Linear, Natural-RF")
    print("Scenarios: Linear, non-linear, interactions, and edge cases")
    
    # Run comparison
    print("\nRunning all methods on all scenarios...")
    results_df = run_comprehensive_comparison(n_samples=2000)
    
    # Save detailed results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'outputs', 'tables')
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(output_dir, 'comprehensive_results.csv'), 
                      index=False)
    print(f"\nDetailed results saved to outputs/tables/comprehensive_results.csv")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Performance heatmap
    fig1 = create_performance_heatmap(results_df)
    
    # Scenario-specific plots
    fig2 = create_scenario_specific_plots(results_df)
    
    # Save figures
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'outputs', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(fig_dir, 'performance_heatmap.png'), 
                 dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(fig_dir, 'scenario_specific_performance.png'),
                 dpi=150, bbox_inches='tight')
    
    print("Visualizations saved to outputs/figures/")
    
    # Create summary
    summary_df = create_summary_table(results_df)
    summary_df.to_csv(os.path.join(output_dir, 'summary_findings.csv'), 
                      index=False)
    
    # Final recommendations
    print("\n" + "="*80)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*80)
    print("""
1. START SIMPLE:
   - Always begin with traditional mediation analysis
   - Check linearity assumptions with diagnostic plots
   - If linear, all methods will give similar results

2. USE DML WHEN:
   - Non-linear relationships are suspected
   - You have enough data for cross-fitting (n > 500)
   - You want robust estimates with ML flexibility

3. USE NATURAL EFFECTS WHEN:
   - Treatment-mediator interactions exist
   - You need clear causal interpretation
   - CDE varies with mediator level

4. WATCH OUT FOR:
   - Near-zero direct/indirect effects (PoMA unstable)
   - Suppression effects (sign reversals)
   - Small sample sizes (ML methods need data)

5. ALWAYS:
   - Report multiple methods for robustness
   - Include confidence intervals
   - Check model diagnostics
   - Visualize relationships before analysis
""")
    
    plt.show()


if __name__ == "__main__":
    main()