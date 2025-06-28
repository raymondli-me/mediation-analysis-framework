"""
Frisch-Waugh-Lovell (FWL) Approach to Mediation

The FWL theorem shows that coefficients from multivariate regression can be obtained
by a two-step residualization process. For mediation:

1. Regress Y on M to get residuals e_Y
2. Regress X on M to get residuals e_X  
3. Regress e_Y on e_X to get the direct effect

This is mathematically equivalent to the coefficient on X from Y ~ X + M.

References:
- Frisch, R., & Waugh, F. V. (1933). Partial time regressions as compared with individual trends
- Lovell, M. C. (1963). Seasonal adjustment of economic time series
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional, Callable, Tuple
import warnings


def fwl_regression(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                  fit_intercept: bool = True) -> Dict:
    """
    Estimate direct effect using Frisch-Waugh-Lovell theorem.
    
    The FWL theorem states that β₁ from Y = β₀ + β₁X + β₂M + ε
    equals the coefficient from regressing (Y - Ŷ|M) on (X - X̂|M).
    
    Parameters
    ----------
    X : np.ndarray
        Treatment variable
    M : np.ndarray  
        Mediator variable
    Y : np.ndarray
        Outcome variable
    fit_intercept : bool
        Whether to fit intercepts in regressions
        
    Returns
    -------
    dict
        Results including direct effect via FWL
    """
    # Reshape for sklearn
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Total effect (for comparison)
    lr_total = LinearRegression(fit_intercept=fit_intercept)
    lr_total.fit(X, Y)
    total_effect = lr_total.coef_[0, 0]
    
    # FWL Step 1: Residualize Y with respect to M
    lr_y_m = LinearRegression(fit_intercept=fit_intercept)
    lr_y_m.fit(M, Y)
    Y_hat_m = lr_y_m.predict(M)
    e_Y = Y.ravel() - Y_hat_m.ravel()
    
    # FWL Step 2: Residualize X with respect to M
    lr_x_m = LinearRegression(fit_intercept=fit_intercept)
    lr_x_m.fit(M, X)
    X_hat_m = lr_x_m.predict(M)
    e_X = X.ravel() - X_hat_m.ravel()
    
    # FWL Step 3: Regress residuals
    if np.var(e_X) < 1e-10:
        warnings.warn("X residuals have near-zero variance. FWL may be unstable.")
        direct_effect_fwl = 0.0
    else:
        lr_residual = LinearRegression(fit_intercept=False)  # No intercept for residuals
        lr_residual.fit(e_X.reshape(-1, 1), e_Y)
        direct_effect_fwl = lr_residual.coef_[0]
    
    # For comparison: traditional multivariate regression
    XM = np.hstack([X, M])
    lr_traditional = LinearRegression(fit_intercept=fit_intercept)
    lr_traditional.fit(XM, Y)
    direct_effect_traditional = lr_traditional.coef_[0, 0]
    
    # Calculate effects
    indirect_effect = total_effect - direct_effect_fwl
    poma = indirect_effect / total_effect if abs(total_effect) > 1e-10 else np.nan
    
    # They should be identical (up to numerical precision)
    difference = abs(direct_effect_fwl - direct_effect_traditional)
    if difference > 1e-10:
        warnings.warn(
            f"FWL ({direct_effect_fwl:.6f}) and traditional "
            f"({direct_effect_traditional:.6f}) give different results. "
            f"Difference: {difference:.2e}"
        )
    
    return {
        'method': 'Frisch-Waugh-Lovell',
        'total_effect': total_effect,
        'direct_effect': direct_effect_fwl,
        'direct_effect_traditional': direct_effect_traditional,
        'indirect_effect': indirect_effect,
        'poma': poma,
        'residual_variance_Y': np.var(e_Y),
        'residual_variance_X': np.var(e_X),
        'r_squared_Y_M': lr_y_m.score(M, Y),
        'r_squared_X_M': lr_x_m.score(M, X),
        'fwl_traditional_difference': difference
    }


def fwl_with_ml(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                ml_model_y: Optional[Callable] = None,
                ml_model_x: Optional[Callable] = None) -> Dict:
    """
    FWL approach using machine learning for residualization.
    
    This extends FWL to non-linear relationships by using ML models
    for the residualization steps.
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    ml_model_y : sklearn-like model or None
        Model for Y|M prediction (default: LinearRegression)
    ml_model_x : sklearn-like model or None
        Model for X|M prediction (default: LinearRegression)
        
    Returns
    -------
    dict
        Results including ML-based FWL estimates
    """
    from sklearn.base import clone
    
    # Default to linear if not specified
    if ml_model_y is None:
        ml_model_y = LinearRegression()
    if ml_model_x is None:
        ml_model_x = LinearRegression()
    
    # Clone models to avoid modifying originals
    model_y = clone(ml_model_y)
    model_x = clone(ml_model_x)
    
    # Reshape
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Total effect (still linear for interpretability)
    lr_total = LinearRegression()
    lr_total.fit(X, Y)
    total_effect = lr_total.coef_[0, 0]
    
    # ML-based residualization
    # Step 1: Y|M
    model_y.fit(M, Y.ravel())
    Y_hat_m = model_y.predict(M)
    e_Y = Y.ravel() - Y_hat_m
    
    # Step 2: X|M
    model_x.fit(M, X.ravel())
    X_hat_m = model_x.predict(M)
    e_X = X.ravel() - X_hat_m
    
    # Step 3: Linear regression on residuals
    if np.var(e_X) < 1e-10:
        direct_effect = 0.0
        warnings.warn("X residuals have near-zero variance after ML residualization.")
    else:
        lr_residual = LinearRegression(fit_intercept=False)
        lr_residual.fit(e_X.reshape(-1, 1), e_Y)
        direct_effect = lr_residual.coef_[0]
    
    indirect_effect = total_effect - direct_effect
    poma = indirect_effect / total_effect if abs(total_effect) > 1e-10 else np.nan
    
    # Calculate R² for ML models
    r2_y_m = 1 - np.mean((Y.ravel() - Y_hat_m)**2) / np.var(Y)
    r2_x_m = 1 - np.mean((X.ravel() - X_hat_m)**2) / np.var(X)
    
    return {
        'method': 'FWL with ML',
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'indirect_effect': indirect_effect,
        'poma': poma,
        'residual_variance_Y': np.var(e_Y),
        'residual_variance_X': np.var(e_X),
        'r_squared_Y_M': r2_y_m,
        'r_squared_X_M': r2_x_m,
        'ml_model_y': type(model_y).__name__,
        'ml_model_x': type(model_x).__name__
    }


def demonstrate_fwl_theorem(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    """
    Demonstrate that FWL gives identical results to traditional regression.
    
    This function shows step-by-step that the coefficient on X from Y ~ X + M
    equals the coefficient from regressing residuals.
    """
    print("="*60)
    print("DEMONSTRATING FRISCH-WAUGH-LOVELL THEOREM")
    print("="*60)
    
    # Traditional approach
    print("\n1. TRADITIONAL APPROACH: Y ~ β₀ + β₁X + β₂M")
    X_2d = X.reshape(-1, 1) if X.ndim == 1 else X
    M_2d = M.reshape(-1, 1) if M.ndim == 1 else M
    Y_2d = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    XM = np.hstack([X_2d, M_2d])
    lr = LinearRegression()
    lr.fit(XM, Y_2d)
    beta_1_traditional = lr.coef_[0, 0]
    beta_2_traditional = lr.coef_[0, 1]
    
    print(f"   β₁ (direct effect) = {beta_1_traditional:.6f}")
    print(f"   β₂ (M→Y effect) = {beta_2_traditional:.6f}")
    
    # FWL approach
    print("\n2. FWL APPROACH:")
    
    # Step 1
    print("\n   Step 1: Regress Y on M")
    lr_y_m = LinearRegression()
    lr_y_m.fit(M_2d, Y_2d)
    Y_hat = lr_y_m.predict(M_2d)
    e_Y = Y_2d.ravel() - Y_hat.ravel()
    print(f"   R²(Y|M) = {lr_y_m.score(M_2d, Y_2d):.4f}")
    print(f"   Residual variance = {np.var(e_Y):.4f}")
    
    # Step 2
    print("\n   Step 2: Regress X on M")
    lr_x_m = LinearRegression()
    lr_x_m.fit(M_2d, X_2d)
    X_hat = lr_x_m.predict(M_2d)
    e_X = X_2d.ravel() - X_hat.ravel()
    print(f"   R²(X|M) = {lr_x_m.score(M_2d, X_2d):.4f}")
    print(f"   Residual variance = {np.var(e_X):.4f}")
    
    # Step 3
    print("\n   Step 3: Regress e_Y on e_X")
    lr_residual = LinearRegression(fit_intercept=False)
    lr_residual.fit(e_X.reshape(-1, 1), e_Y)
    beta_1_fwl = lr_residual.coef_[0]
    print(f"   Coefficient = {beta_1_fwl:.6f}")
    
    # Comparison
    print("\n3. COMPARISON:")
    print(f"   Traditional β₁: {beta_1_traditional:.6f}")
    print(f"   FWL β₁:        {beta_1_fwl:.6f}")
    print(f"   Difference:     {abs(beta_1_traditional - beta_1_fwl):.2e}")
    
    if abs(beta_1_traditional - beta_1_fwl) < 1e-10:
        print("\n✓ THEOREM VERIFIED: FWL gives identical results!")
    else:
        print("\n✗ WARNING: Results differ more than expected")
    
    # Intuition
    print("\n4. INTUITION:")
    print("   FWL removes the linear influence of M from both Y and X,")
    print("   then looks at the relationship in the 'leftover' variation.")
    print("   This isolates the direct X→Y effect controlling for M.")


def fwl_path_analysis(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> Dict:
    """
    Detailed path analysis using FWL approach.
    
    Returns all path coefficients and residual correlations.
    """
    # Get basic FWL results
    results = fwl_regression(X, M, Y)
    
    # Additional path analysis
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Path a: X → M
    lr_a = LinearRegression()
    lr_a.fit(X, M)
    path_a = lr_a.coef_[0, 0]
    
    # Path b: M → Y (controlling for X)
    XM = np.hstack([X, M])
    lr_b = LinearRegression()
    lr_b.fit(XM, Y)
    path_b = lr_b.coef_[0, 1]
    
    # Get residuals for correlation analysis
    lr_y_m = LinearRegression()
    lr_y_m.fit(M, Y)
    e_Y = Y.ravel() - lr_y_m.predict(M).ravel()
    
    lr_x_m = LinearRegression()
    lr_x_m.fit(M, X)
    e_X = X.ravel() - lr_x_m.predict(M).ravel()
    
    # Residual correlation
    if np.var(e_X) > 1e-10 and np.var(e_Y) > 1e-10:
        residual_correlation = np.corrcoef(e_X, e_Y)[0, 1]
    else:
        residual_correlation = 0.0
    
    return {
        **results,
        'path_a': path_a,
        'path_b': path_b,
        'indirect_via_paths': path_a * path_b,
        'residual_correlation': residual_correlation,
        'proportion_X_explained_by_M': lr_x_m.score(M, X),
        'proportion_Y_explained_by_M': lr_y_m.score(M, Y)
    }