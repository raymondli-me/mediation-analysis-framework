"""
Double Machine Learning (DML) Approach to Mediation

DML extends the FWL approach by:
1. Using machine learning for flexible nuisance function estimation
2. Cross-fitting to avoid overfitting bias
3. Providing the reduction formula for analytical computation

This implementation includes:
- Standard DML with cross-fitting
- The reduction formula from the paper
- Comparison between empirical and formula-based estimates

References:
- Chernozhukov et al. (2018). Double/debiased machine learning
- DML-Mediation paper's reduction formula
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from typing import Dict, Optional, Union, Tuple
import warnings


def dml_mediation(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                 ml_model_y: Optional[BaseEstimator] = None,
                 ml_model_x: Optional[BaseEstimator] = None,
                 n_folds: int = 5,
                 random_state: int = 42) -> Dict:
    """
    Estimate mediation effects using Double Machine Learning.
    
    DML procedure:
    1. Split data into K folds
    2. For each fold k:
       - Train ML models on other folds
       - Predict Y|M and X|M on fold k
       - Calculate residuals
    3. Regress residuals to get direct effect
    
    Parameters
    ----------
    X : np.ndarray
        Treatment variable
    M : np.ndarray
        Mediator variable  
    Y : np.ndarray
        Outcome variable
    ml_model_y : sklearn estimator or None
        Model for Y|M (default: LinearRegression)
    ml_model_x : sklearn estimator or None
        Model for X|M (default: LinearRegression)
    n_folds : int
        Number of cross-fitting folds
    random_state : int
        Random seed for fold splitting
        
    Returns
    -------
    dict
        DML estimates and diagnostics
    """
    # Default to linear models if not specified
    if ml_model_y is None:
        ml_model_y = LinearRegression()
    if ml_model_x is None:
        ml_model_x = LinearRegression()
    
    # Ensure proper shapes
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    n = len(X)
    
    # Total effect (using simple linear regression)
    lr_total = LinearRegression()
    lr_total.fit(X, Y)
    total_effect = lr_total.coef_[0, 0]
    
    # Cross-fitting
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Initialize arrays for out-of-fold predictions
    Y_hat = np.zeros(n)
    X_hat = np.zeros(n)
    
    # Store fold-specific models for diagnostics
    fold_models_y = []
    fold_models_x = []
    
    for train_idx, test_idx in kf.split(X):
        # Clone models for this fold
        model_y_fold = clone(ml_model_y)
        model_x_fold = clone(ml_model_x)
        
        # Train on training fold
        model_y_fold.fit(M[train_idx], Y[train_idx].ravel())
        model_x_fold.fit(M[train_idx], X[train_idx].ravel())
        
        # Predict on test fold (out-of-fold predictions)
        Y_hat[test_idx] = model_y_fold.predict(M[test_idx])
        X_hat[test_idx] = model_x_fold.predict(M[test_idx])
        
        # Store models
        fold_models_y.append(model_y_fold)
        fold_models_x.append(model_x_fold)
    
    # Calculate residuals
    e_Y = Y.ravel() - Y_hat
    e_X = X.ravel() - X_hat
    
    # Estimate direct effect from residuals
    if np.var(e_X) < 1e-10:
        warnings.warn("X residuals have near-zero variance. DML may be unstable.")
        direct_effect = 0.0
        se_direct = np.nan
    else:
        # Method 1: Linear regression
        lr_residual = LinearRegression(fit_intercept=False)
        lr_residual.fit(e_X.reshape(-1, 1), e_Y)
        direct_effect = lr_residual.coef_[0]
        
        # Method 2: Simple covariance (should be identical)
        direct_effect_cov = np.cov(e_Y, e_X)[0, 1] / np.var(e_X)
        
        # Standard error (approximate)
        n_eff = len(e_Y)
        sigma2 = np.mean(e_Y**2)
        se_direct = np.sqrt(sigma2 / (n_eff * np.var(e_X)))
    
    # Calculate indirect effect and PoMA
    indirect_effect = total_effect - direct_effect
    poma = indirect_effect / total_effect if abs(total_effect) > 1e-10 else np.nan
    
    # Model fit statistics
    r2_y_given_m = 1 - np.mean((Y.ravel() - Y_hat)**2) / np.var(Y)
    r2_x_given_m = 1 - np.mean((X.ravel() - X_hat)**2) / np.var(X)
    
    # Also calculate using the reduction formula
    formula_results = reduction_formula(X, Y, M, Y_hat, X_hat, e_Y, e_X)
    
    return {
        'method': 'Double Machine Learning',
        'total_effect': total_effect,
        'direct_effect': direct_effect,
        'direct_effect_cov': direct_effect_cov,
        'indirect_effect': indirect_effect,
        'poma': poma,
        'poma_formula': formula_results['poma_formula'],
        'se_direct': se_direct,
        'n_folds': n_folds,
        'r2_y_given_m': r2_y_given_m,
        'r2_x_given_m': r2_x_given_m,
        'residual_var_Y': np.var(e_Y),
        'residual_var_X': np.var(e_X),
        'ml_model_y': type(ml_model_y).__name__,
        'ml_model_x': type(ml_model_x).__name__,
        'formula_results': formula_results
    }


def reduction_formula(X: np.ndarray, Y: np.ndarray, M: np.ndarray,
                     Y_hat: np.ndarray, X_hat: np.ndarray,
                     e_Y: np.ndarray, e_X: np.ndarray) -> Dict:
    """
    Calculate direct effect using the reduction formula.
    
    The formula is:
    c'/c = [1 - Cov(Ŷ,X̂)/Cov(Y,X) - C1 - C2] / [1 - Var(X̂)/Var(X) - C3]
    
    Where:
    - C1 = Cov(e_Y, X̂) / Cov(Y,X)
    - C2 = Cov(e_X, Ŷ) / Cov(Y,X)  
    - C3 = 2 * Cov(e_X, X̂) / Var(X)
    
    Parameters
    ----------
    X, Y, M : np.ndarray
        Original data
    Y_hat, X_hat : np.ndarray
        Predictions from M
    e_Y, e_X : np.ndarray
        Residuals
        
    Returns
    -------
    dict
        Formula components and final estimate
    """
    # Ensure 1D arrays for covariance calculations
    X = X.ravel()
    Y = Y.ravel()
    Y_hat = Y_hat.ravel()
    X_hat = X_hat.ravel()
    e_Y = e_Y.ravel()
    e_X = e_X.ravel()
    
    # Basic covariances and variances
    cov_YX = np.cov(Y, X)[0, 1]
    var_X = np.var(X, ddof=1)
    cov_Yhat_Xhat = np.cov(Y_hat, X_hat)[0, 1]
    var_Xhat = np.var(X_hat, ddof=1)
    
    # Correction terms
    if abs(cov_YX) > 1e-10:
        C1 = np.cov(e_Y, X_hat)[0, 1] / cov_YX
        C2 = np.cov(e_X, Y_hat)[0, 1] / cov_YX
    else:
        C1 = C2 = 0.0
        warnings.warn("Cov(Y,X) near zero. Reduction formula unstable.")
    
    C3 = 2 * np.cov(e_X, X_hat)[0, 1] / var_X if var_X > 1e-10 else 0.0
    
    # Calculate numerator and denominator
    if abs(cov_YX) > 1e-10:
        ratio_cov = cov_Yhat_Xhat / cov_YX
    else:
        ratio_cov = 0.0
    
    ratio_var = var_Xhat / var_X if var_X > 1e-10 else 0.0
    
    numerator = 1 - ratio_cov - C1 - C2
    denominator = 1 - ratio_var - C3
    
    # Calculate c'/c ratio
    if abs(denominator) > 1e-10:
        direct_total_ratio = numerator / denominator
    else:
        direct_total_ratio = np.nan
        warnings.warn("Denominator near zero in reduction formula.")
    
    # Get total effect for final calculation
    lr = LinearRegression()
    lr.fit(X.reshape(-1, 1), Y)
    total_effect = lr.coef_[0]
    
    # Direct effect from formula
    direct_effect_formula = direct_total_ratio * total_effect if not np.isnan(direct_total_ratio) else np.nan
    
    # PoMA from formula
    poma_formula = 1 - direct_total_ratio if not np.isnan(direct_total_ratio) else np.nan
    
    return {
        'cov_YX': cov_YX,
        'var_X': var_X,
        'cov_Yhat_Xhat': cov_Yhat_Xhat,
        'var_Xhat': var_Xhat,
        'C1': C1,
        'C2': C2,
        'C3': C3,
        'ratio_cov': ratio_cov,
        'ratio_var': ratio_var,
        'numerator': numerator,
        'denominator': denominator,
        'direct_total_ratio': direct_total_ratio,
        'direct_effect_formula': direct_effect_formula,
        'poma_formula': poma_formula,
        'total_effect': total_effect
    }


def dml_with_confidence_intervals(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                                 ml_model_y: Optional[BaseEstimator] = None,
                                 ml_model_x: Optional[BaseEstimator] = None,
                                 n_folds: int = 5,
                                 n_bootstrap: int = 100,
                                 confidence_level: float = 0.95,
                                 random_state: int = 42) -> Dict:
    """
    DML with bootstrap confidence intervals.
    
    Uses the bootstrap to get confidence intervals for the direct effect,
    accounting for the uncertainty in the ML predictions.
    """
    # Original estimate
    original = dml_mediation(X, M, Y, ml_model_y, ml_model_x, n_folds, random_state)
    
    # Bootstrap
    n = len(X)
    direct_effects = []
    pomas = []
    
    for b in range(n_bootstrap):
        # Resample
        idx = np.random.RandomState(random_state + b).choice(n, n, replace=True)
        X_boot = X[idx]
        M_boot = M[idx]
        Y_boot = Y[idx]
        
        try:
            boot_result = dml_mediation(X_boot, M_boot, Y_boot, 
                                      ml_model_y, ml_model_x, 
                                      n_folds, random_state + b)
            direct_effects.append(boot_result['direct_effect'])
            pomas.append(boot_result['poma'])
        except:
            continue
    
    # Calculate CIs
    alpha = 1 - confidence_level
    direct_effects = np.array(direct_effects)
    pomas = np.array([p for p in pomas if not np.isnan(p)])
    
    ci_direct = np.percentile(direct_effects, [100*alpha/2, 100*(1-alpha/2)])
    ci_poma = np.percentile(pomas, [100*alpha/2, 100*(1-alpha/2)]) if len(pomas) > 0 else [np.nan, np.nan]
    
    return {
        **original,
        'ci_direct': ci_direct,
        'ci_poma': ci_poma,
        'se_direct_bootstrap': np.std(direct_effects),
        'se_poma_bootstrap': np.std(pomas) if len(pomas) > 0 else np.nan,
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }


def compare_dml_implementations(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> None:
    """
    Compare different DML implementations and the reduction formula.
    """
    print("="*70)
    print("COMPARING DML IMPLEMENTATIONS")
    print("="*70)
    
    # Linear DML (should match FWL exactly)
    print("\n1. DML with Linear Models:")
    linear_result = dml_mediation(X, M, Y, n_folds=5)
    print(f"   Direct effect (empirical): {linear_result['direct_effect']:.6f}")
    print(f"   Direct effect (formula):   {linear_result['formula_results']['direct_effect_formula']:.6f}")
    print(f"   PoMA (empirical):         {linear_result['poma']:.4f}")
    print(f"   PoMA (formula):           {linear_result['poma_formula']:.4f}")
    
    # ML DML
    print("\n2. DML with XGBoost:")
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        ml_result = dml_mediation(X, M, Y, 
                                ml_model_y=xgb_model,
                                ml_model_x=xgb_model,
                                n_folds=5)
        print(f"   Direct effect (empirical): {ml_result['direct_effect']:.6f}")
        print(f"   Direct effect (formula):   {ml_result['formula_results']['direct_effect_formula']:.6f}")
        print(f"   PoMA (empirical):         {ml_result['poma']:.4f}")
        print(f"   PoMA (formula):           {ml_result['poma_formula']:.4f}")
        print(f"   R²(Y|M):                  {ml_result['r2_y_given_m']:.4f}")
        print(f"   R²(X|M):                  {ml_result['r2_x_given_m']:.4f}")
    except ImportError:
        print("   [XGBoost not available]")
    
    # Formula components
    print("\n3. Reduction Formula Components:")
    formula = linear_result['formula_results']
    print(f"   Cov(Y,X):        {formula['cov_YX']:.4f}")
    print(f"   Cov(Ŷ,X̂):       {formula['cov_Yhat_Xhat']:.4f}")
    print(f"   Ratio:           {formula['ratio_cov']:.4f}")
    print(f"   C1:              {formula['C1']:.4f}")
    print(f"   C2:              {formula['C2']:.4f}")
    print(f"   C3:              {formula['C3']:.4f}")
    print(f"   Numerator:       {formula['numerator']:.4f}")
    print(f"   Denominator:     {formula['denominator']:.4f}")
    
    # Validation
    print("\n4. Validation:")
    emp_formula_diff = abs(linear_result['direct_effect'] - 
                          linear_result['formula_results']['direct_effect_formula'])
    print(f"   |Empirical - Formula|: {emp_formula_diff:.2e}")
    
    if emp_formula_diff < 1e-6:
        print("   ✓ Reduction formula matches empirical estimate!")
    else:
        print("   ✗ Warning: Formula and empirical estimates differ")