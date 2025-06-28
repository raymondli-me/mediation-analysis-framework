"""
Traditional Mediation Analysis (Baron & Kenny approach)

This module implements the classic Baron & Kenny (1986) approach to mediation analysis.
It estimates:
- Total effect: c from Y = c*X + e
- Direct effect: c' from Y = c'*X + b*M + e  
- Indirect effect: a*b (or c - c')

References:
- Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, Optional
import warnings


def estimate_effects(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                    return_models: bool = False) -> Dict:
    """
    Estimate mediation effects using traditional Baron & Kenny approach.
    
    Parameters
    ----------
    X : np.ndarray
        Treatment/exposure variable (n,) or (n, 1)
    M : np.ndarray
        Mediator variable (n,) or (n, 1)
    Y : np.ndarray
        Outcome variable (n,) or (n, 1)
    return_models : bool
        Whether to return fitted model objects
    
    Returns
    -------
    dict
        Dictionary containing:
        - total_effect: Total effect of X on Y
        - direct_effect: Direct effect of X on Y controlling for M
        - indirect_effect: Indirect effect through M
        - poma: Proportion of mediation accuracy
        - path_a: X→M coefficient
        - path_b: M→Y coefficient controlling for X
        - models: Dict of fitted models (if return_models=True)
    """
    # Ensure 2D arrays for sklearn
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Step 1: Total effect (c path)
    # Y = c*X + e
    model_total = LinearRegression()
    model_total.fit(X, Y)
    total_effect = model_total.coef_[0, 0]
    
    # Step 2: X → M path (a path)
    # M = a*X + e
    model_a = LinearRegression()
    model_a.fit(X, M)
    path_a = model_a.coef_[0, 0]
    
    # Step 3: Direct effect (c' path) and M → Y effect (b path)
    # Y = c'*X + b*M + e
    XM = np.hstack([X, M])
    model_direct = LinearRegression()
    model_direct.fit(XM, Y)
    direct_effect = model_direct.coef_[0, 0]  # c'
    path_b = model_direct.coef_[0, 1]        # b
    
    # Step 4: Calculate indirect effect
    # Two ways to calculate (should be nearly identical):
    indirect_effect_prod = path_a * path_b      # Product method
    indirect_effect_diff = total_effect - direct_effect  # Difference method
    
    # Use difference method as primary (more stable)
    indirect_effect = indirect_effect_diff
    
    # Check consistency
    if abs(indirect_effect_prod - indirect_effect_diff) > 0.01:
        warnings.warn(
            f"Product ({indirect_effect_prod:.4f}) and difference "
            f"({indirect_effect_diff:.4f}) methods give different results. "
            "Model may be misspecified."
        )
    
    # Calculate PoMA
    poma = calculate_poma(total_effect, direct_effect)
    
    result = {
        'method': 'Traditional (Baron & Kenny)',
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'indirect_effect_prod': indirect_effect_prod,
        'poma': poma,
        'path_a': path_a,
        'path_b': path_b,
    }
    
    if return_models:
        result['models'] = {
            'total': model_total,
            'x_to_m': model_a,
            'direct': model_direct
        }
    
    return result


def calculate_poma(total_effect: float, direct_effect: float,
                  min_effect: float = 1e-10) -> float:
    """
    Calculate Proportion of Mediation Accuracy (PoMA).
    
    PoMA = 1 - (direct_effect / total_effect) = indirect_effect / total_effect
    
    Parameters
    ----------
    total_effect : float
        Total effect of X on Y
    direct_effect : float
        Direct effect of X on Y controlling for M
    min_effect : float
        Minimum absolute total effect to avoid division by zero
        
    Returns
    -------
    float
        PoMA value, or NaN if total effect is too small
    """
    if abs(total_effect) < min_effect:
        warnings.warn(
            f"Total effect ({total_effect:.4f}) is near zero. "
            "PoMA is undefined/unstable."
        )
        return np.nan
    
    poma = 1 - (direct_effect / total_effect)
    
    # Check for unusual values
    if poma < -0.5 or poma > 1.5:
        warnings.warn(
            f"PoMA = {poma:.2f} is outside typical range [0, 1]. "
            "Possible suppression or inconsistent mediation."
        )
    
    return poma


def baron_kenny_steps(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                     alpha: float = 0.05) -> Dict:
    """
    Perform Baron & Kenny's 4-step mediation test.
    
    Steps:
    1. Test if X → Y (total effect significant)
    2. Test if X → M (path a significant)
    3. Test if M → Y controlling for X (path b significant)
    4. Test if direct effect < total effect
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Test results for each step
    """
    from scipy import stats
    
    # Get effects
    results = estimate_effects(X, M, Y, return_models=True)
    models = results['models']
    
    # Helper function for t-test
    def get_pvalue(model, X, y, coef_idx=0):
        """Get p-value for coefficient"""
        n = len(y)
        k = X.shape[1]
        
        # Residual sum of squares
        y_pred = model.predict(X)
        rss = np.sum((y.ravel() - y_pred)**2)
        
        # Standard error
        mse = rss / (n - k)
        var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
        se = np.sqrt(var_coef[coef_idx])
        
        # t-statistic
        t_stat = model.coef_.ravel()[coef_idx] / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))
        
        return p_value, se, t_stat
    
    # Reshape for consistency
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Step 1: Total effect
    p_total, se_total, t_total = get_pvalue(models['total'], X, Y)
    step1 = {
        'significant': p_total < alpha,
        'effect': results['total_effect'],
        'p_value': p_total,
        'conclusion': 'X affects Y' if p_total < alpha else 'No total effect'
    }
    
    # Step 2: X → M
    p_a, se_a, t_a = get_pvalue(models['x_to_m'], X, M)
    step2 = {
        'significant': p_a < alpha,
        'effect': results['path_a'],
        'p_value': p_a,
        'conclusion': 'X affects M' if p_a < alpha else 'No effect on mediator'
    }
    
    # Step 3: M → Y controlling for X
    XM = np.hstack([X, M])
    p_b, se_b, t_b = get_pvalue(models['direct'], XM, Y, coef_idx=1)
    step3 = {
        'significant': p_b < alpha,
        'effect': results['path_b'],
        'p_value': p_b,
        'conclusion': 'M affects Y' if p_b < alpha else 'Mediator does not affect outcome'
    }
    
    # Step 4: Compare direct to total
    reduction = results['total_effect'] - results['direct_effect']
    step4 = {
        'reduction': reduction,
        'percent_reduction': (reduction / results['total_effect'] * 100 
                            if results['total_effect'] != 0 else np.nan),
        'conclusion': ('Partial mediation' if results['direct_effect'] != 0 
                      else 'Complete mediation')
    }
    
    # Overall conclusion
    all_significant = step1['significant'] and step2['significant'] and step3['significant']
    
    return {
        'step1_total_effect': step1,
        'step2_x_to_m': step2,
        'step3_m_to_y': step3,
        'step4_reduction': step4,
        'mediation_detected': all_significant,
        'overall_conclusion': (
            'Significant mediation detected' if all_significant 
            else 'No significant mediation'
        ),
        'poma': results['poma']
    }


def bootstrap_indirect_effect(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95,
                            method: str = 'percentile') -> Dict:
    """
    Bootstrap confidence interval for indirect effect.
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    method : str
        'percentile' or 'bca' (bias-corrected and accelerated)
        
    Returns
    -------
    dict
        Bootstrap results including CI
    """
    n = len(X)
    indirect_effects = []
    
    # Original estimate
    original = estimate_effects(X, M, Y)
    
    # Bootstrap
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, n, replace=True)
        X_boot = X[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]
        
        # Estimate on bootstrap sample
        try:
            boot_result = estimate_effects(X_boot, M_boot, Y_boot)
            indirect_effects.append(boot_result['indirect_effect'])
        except:
            # Skip if estimation fails
            continue
    
    indirect_effects = np.array(indirect_effects)
    
    # Calculate CI
    alpha = 1 - confidence_level
    if method == 'percentile':
        ci_lower = np.percentile(indirect_effects, 100 * alpha/2)
        ci_upper = np.percentile(indirect_effects, 100 * (1 - alpha/2))
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
    return {
        'indirect_effect': original['indirect_effect'],
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'n_bootstrap': n_bootstrap,
        'bootstrap_se': np.std(indirect_effects),
        'significant': not (ci_lower <= 0 <= ci_upper)
    }