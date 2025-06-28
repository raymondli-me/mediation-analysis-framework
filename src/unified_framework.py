"""
Unified Framework for Mediation Analysis

Provides a single interface to run all mediation methods and compare results.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import warnings

# Import all methods
from .traditional_mediation import estimate_effects as traditional_estimate
from .fwl_approach import fwl_regression, fwl_with_ml
from .dml_approach import dml_mediation
from .causal_mediation import estimate_natural_effects


def run_all_methods(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                   ml_models: Optional[Dict[str, Any]] = None,
                   include_natural_effects: bool = True,
                   n_folds: int = 5,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Run all mediation analysis methods and return comparison.
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    ml_models : dict or None
        ML models for DML {'y': model, 'x': model}
    include_natural_effects : bool
        Whether to compute natural effects (slower)
    n_folds : int
        Cross-fitting folds for DML
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Results from all methods
    """
    results = []
    
    # 1. Traditional mediation
    try:
        trad = traditional_estimate(X, M, Y)
        results.append({
            'Method': 'Traditional',
            'Total Effect': trad['total_effect'],
            'Direct Effect': trad['direct_effect'],
            'Indirect Effect': trad['indirect_effect'],
            'PoMA': trad['poma'],
            'Notes': ''
        })
    except Exception as e:
        results.append({
            'Method': 'Traditional',
            'Total Effect': np.nan,
            'Direct Effect': np.nan,
            'Indirect Effect': np.nan,
            'PoMA': np.nan,
            'Notes': f'Error: {str(e)}'
        })
    
    # 2. FWL approach
    try:
        fwl = fwl_regression(X, M, Y)
        results.append({
            'Method': 'FWL',
            'Total Effect': fwl['total_effect'],
            'Direct Effect': fwl['direct_effect'],
            'Indirect Effect': fwl['indirect_effect'],
            'PoMA': fwl['poma'],
            'Notes': f"Diff from trad: {fwl['fwl_traditional_difference']:.2e}"
        })
    except Exception as e:
        results.append({
            'Method': 'FWL',
            'Total Effect': np.nan,
            'Direct Effect': np.nan,
            'Indirect Effect': np.nan,
            'PoMA': np.nan,
            'Notes': f'Error: {str(e)}'
        })
    
    # 3. DML with linear models
    try:
        dml_linear = dml_mediation(X, M, Y, n_folds=n_folds, random_state=random_state)
        results.append({
            'Method': 'DML (Linear)',
            'Total Effect': dml_linear['total_effect'],
            'Direct Effect': dml_linear['direct_effect'],
            'Indirect Effect': dml_linear['indirect_effect'],
            'PoMA': dml_linear['poma'],
            'Notes': f"Formula PoMA: {dml_linear['poma_formula']:.4f}"
        })
    except Exception as e:
        results.append({
            'Method': 'DML (Linear)',
            'Total Effect': np.nan,
            'Direct Effect': np.nan,
            'Indirect Effect': np.nan,
            'PoMA': np.nan,
            'Notes': f'Error: {str(e)}'
        })
    
    # 4. DML with ML models (if provided)
    if ml_models is not None:
        try:
            dml_ml = dml_mediation(
                X, M, Y, 
                ml_model_y=ml_models.get('y'),
                ml_model_x=ml_models.get('x'),
                n_folds=n_folds,
                random_state=random_state
            )
            results.append({
                'Method': f"DML ({type(ml_models['y']).__name__})",
                'Total Effect': dml_ml['total_effect'],
                'Direct Effect': dml_ml['direct_effect'],
                'Indirect Effect': dml_ml['indirect_effect'],
                'PoMA': dml_ml['poma'],
                'Notes': f"R²(Y|M): {dml_ml['r2_y_given_m']:.3f}"
            })
        except Exception as e:
            results.append({
                'Method': 'DML (ML)',
                'Total Effect': np.nan,
                'Direct Effect': np.nan,
                'Indirect Effect': np.nan,
                'PoMA': np.nan,
                'Notes': f'Error: {str(e)}'
            })
    
    # 5. Natural effects (if requested)
    if include_natural_effects:
        try:
            natural = estimate_natural_effects(X, M, Y, interaction=True)
            results.append({
                'Method': 'Natural Effects',
                'Total Effect': natural['total_effect'],
                'Direct Effect': natural['nde'],
                'Indirect Effect': natural['nie'],
                'PoMA': natural['prop_mediated'],
                'Notes': 'With interaction'
            })
        except Exception as e:
            results.append({
                'Method': 'Natural Effects',
                'Total Effect': np.nan,
                'Direct Effect': np.nan,
                'Indirect Effect': np.nan,
                'PoMA': np.nan,
                'Notes': f'Error: {str(e)}'
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add summary statistics
    if len(df) > 0:
        df['Direct/Total Ratio'] = df['Direct Effect'] / df['Total Effect']
        df['Check Sum'] = df['Direct Effect'] + df['Indirect Effect'] - df['Total Effect']
    
    return df


def compare_methods_visually(results_df: pd.DataFrame, title: str = "Method Comparison"):
    """
    Create visualization comparing methods.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Filter out failed methods
    valid_df = results_df[results_df['Total Effect'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid results to visualize")
        return None
    
    # 1. Effect sizes comparison
    ax = axes[0, 0]
    x = np.arange(len(valid_df))
    width = 0.25
    
    ax.bar(x - width, valid_df['Total Effect'], width, label='Total', alpha=0.8)
    ax.bar(x, valid_df['Direct Effect'], width, label='Direct', alpha=0.8)
    ax.bar(x + width, valid_df['Indirect Effect'], width, label='Indirect', alpha=0.8)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Effect Size')
    ax.set_title('Effect Decomposition by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_df['Method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PoMA comparison
    ax = axes[0, 1]
    ax.bar(valid_df['Method'], valid_df['PoMA'], alpha=0.8, color='green')
    ax.set_xlabel('Method')
    ax.set_ylabel('PoMA')
    ax.set_title('Proportion Mediated')
    ax.set_xticklabels(valid_df['Method'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    
    # 3. Direct/Total ratio
    ax = axes[1, 0]
    ax.bar(valid_df['Method'], valid_df['Direct/Total Ratio'], alpha=0.8, color='orange')
    ax.set_xlabel('Method')
    ax.set_ylabel('Direct/Total Ratio')
    ax.set_title('Direct Effect as Proportion of Total')
    ax.set_xticklabels(valid_df['Method'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    
    # 4. Consistency check
    ax = axes[1, 1]
    ax.bar(valid_df['Method'], valid_df['Check Sum'], alpha=0.8, color='red')
    ax.set_xlabel('Method')
    ax.set_ylabel('Direct + Indirect - Total')
    ax.set_title('Consistency Check (should be ≈ 0)')
    ax.set_xticklabels(valid_df['Method'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    return fig


def diagnostic_summary(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Run diagnostics to recommend appropriate method.
    """
    # Check treatment type
    X_unique = np.unique(X)
    is_binary = len(X_unique) == 2 and set(X_unique).issubset({0, 1})
    
    # Check correlations
    corr_XY = np.corrcoef(X.ravel(), Y.ravel())[0, 1]
    corr_XM = np.corrcoef(X.ravel(), M.ravel())[0, 1]
    corr_MY = np.corrcoef(M.ravel(), Y.ravel())[0, 1]
    
    # Check for potential non-linearity
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    # X-M relationship
    lr_xm = LinearRegression()
    lr_xm.fit(X.reshape(-1, 1), M)
    r2_linear_xm = lr_xm.score(X.reshape(-1, 1), M)
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    lr_xm_poly = LinearRegression()
    lr_xm_poly.fit(X_poly, M)
    r2_poly_xm = lr_xm_poly.score(X_poly, M)
    
    nonlinearity_xm = r2_poly_xm - r2_linear_xm
    
    # M-Y relationship
    lr_my = LinearRegression()
    lr_my.fit(M.reshape(-1, 1), Y)
    r2_linear_my = lr_my.score(M.reshape(-1, 1), Y)
    
    M_poly = poly.fit_transform(M.reshape(-1, 1))
    lr_my_poly = LinearRegression()
    lr_my_poly.fit(M_poly, Y)
    r2_poly_my = lr_my_poly.score(M_poly, Y)
    
    nonlinearity_my = r2_poly_my - r2_linear_my
    
    # Check for interaction
    XM = np.column_stack([X.ravel(), M.ravel(), X.ravel() * M.ravel()])
    lr_int = LinearRegression()
    lr_int.fit(XM, Y)
    interaction_coef = lr_int.coef_[2]
    
    # Standard error approximation
    n = len(Y)
    residuals = Y.ravel() - lr_int.predict(XM)
    mse = np.mean(residuals**2)
    se_approx = np.sqrt(mse / n)
    interaction_significant = abs(interaction_coef) > 2 * se_approx
    
    # Recommendations
    recommendations = []
    
    if abs(corr_XY) < 0.05:
        recommendations.append("⚠️ Very low X-Y correlation. PoMA may be unstable.")
    
    if nonlinearity_xm > 0.05 or nonlinearity_my > 0.05:
        recommendations.append("📊 Non-linear relationships detected. Consider DML with ML models.")
    
    if interaction_significant:
        recommendations.append("🔄 Interaction detected. Natural effects recommended.")
    
    if is_binary:
        recommendations.append("✓ Binary treatment. All methods applicable.")
    else:
        recommendations.append("📈 Continuous treatment. Check linearity assumptions.")
    
    if len(recommendations) == 0:
        recommendations.append("✓ Linear relationships. Traditional methods should work well.")
    
    return {
        'treatment_type': 'binary' if is_binary else 'continuous',
        'correlations': {
            'X-Y': corr_XY,
            'X-M': corr_XM,
            'M-Y': corr_MY
        },
        'nonlinearity': {
            'X-M': nonlinearity_xm,
            'M-Y': nonlinearity_my
        },
        'interaction': {
            'coefficient': interaction_coef,
            'significant': interaction_significant
        },
        'recommendations': recommendations
    }