"""
Causal Mediation Framework - Natural Direct and Indirect Effects

This module implements estimation of Natural Direct Effects (NDE) and 
Natural Indirect Effects (NIE) which provide a causal decomposition
that works even with treatment-mediator interactions.

Key concepts:
- NDE: Effect of treatment holding mediator at its natural value under control
- NIE: Effect through change in mediator
- Works with interactions and non-linearities
- Based on potential outcomes framework

References:
- Pearl, J. (2001). Direct and indirect effects
- VanderWeele, T. J. (2015). Explanation in causal inference
- Imai et al. (2010). A general approach to causal mediation analysis
"""

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Dict, Optional, Union, Callable, Tuple
import warnings


def estimate_natural_effects(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                           outcome_model: Optional[BaseEstimator] = None,
                           mediator_model: Optional[BaseEstimator] = None,
                           interaction: bool = True,
                           bootstrap_ci: bool = False,
                           n_bootstrap: int = 1000,
                           confidence_level: float = 0.95,
                           random_state: int = 42) -> Dict:
    """
    Estimate Natural Direct and Indirect Effects.
    
    For binary X:
    - NDE = E[Y(1,M(0))] - E[Y(0,M(0))]
    - NIE = E[Y(1,M(1))] - E[Y(1,M(0))]
    
    For continuous X (using standardized effects):
    - Effects per unit change in X
    
    Parameters
    ----------
    X : np.ndarray
        Treatment variable (binary or continuous)
    M : np.ndarray
        Mediator variable
    Y : np.ndarray
        Outcome variable
    outcome_model : sklearn estimator or None
        Model for Y|X,M (default: LinearRegression with interaction)
    mediator_model : sklearn estimator or None
        Model for M|X (default: LinearRegression)
    interaction : bool
        Whether to include X*M interaction in outcome model
    bootstrap_ci : bool
        Whether to compute bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level for intervals
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Natural effects estimates and diagnostics
    """
    # Check if treatment is binary
    X_flat = X.ravel()
    is_binary = len(np.unique(X_flat)) == 2 and set(np.unique(X_flat)).issubset({0, 1})
    
    if is_binary:
        return _estimate_natural_effects_binary(
            X, M, Y, outcome_model, mediator_model, 
            interaction, bootstrap_ci, n_bootstrap, 
            confidence_level, random_state
        )
    else:
        return _estimate_natural_effects_continuous(
            X, M, Y, outcome_model, mediator_model,
            interaction, bootstrap_ci, n_bootstrap,
            confidence_level, random_state
        )


def _estimate_natural_effects_binary(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                                   outcome_model: Optional[BaseEstimator] = None,
                                   mediator_model: Optional[BaseEstimator] = None,
                                   interaction: bool = True,
                                   bootstrap_ci: bool = False,
                                   n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95,
                                   random_state: int = 42) -> Dict:
    """
    Natural effects for binary treatment.
    """
    # Default models
    if outcome_model is None:
        outcome_model = LinearRegression()
    if mediator_model is None:
        mediator_model = LinearRegression()
    
    # Ensure proper shapes
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Fit mediator model: M|X
    mediator_model = clone(mediator_model)
    mediator_model.fit(X, M.ravel())
    
    # Get M distributions under X=0 and X=1
    X_0 = X[X.ravel() == 0]
    X_1 = X[X.ravel() == 1]
    M_0 = M[X.ravel() == 0]
    M_1 = M[X.ravel() == 1]
    Y_0 = Y[X.ravel() == 0]
    Y_1 = Y[X.ravel() == 1]
    
    # Fit outcome models
    if interaction:
        # Include interaction term
        XM_int = X * M
        features = np.hstack([X, M, XM_int])
        
        outcome_model = clone(outcome_model)
        outcome_model.fit(features, Y.ravel())
        
        # Helper function for predictions
        def predict_y(x_val, m_val):
            x_arr = np.full_like(m_val, x_val).reshape(-1, 1)
            m_arr = m_val.reshape(-1, 1)
            xm_arr = x_arr * m_arr
            features = np.hstack([x_arr, m_arr, xm_arr])
            return outcome_model.predict(features)
    else:
        # No interaction
        features = np.hstack([X, M])
        outcome_model = clone(outcome_model)
        outcome_model.fit(features, Y.ravel())
        
        def predict_y(x_val, m_val):
            x_arr = np.full_like(m_val, x_val).reshape(-1, 1)
            m_arr = m_val.reshape(-1, 1)
            features = np.hstack([x_arr, m_arr])
            return outcome_model.predict(features)
    
    # Calculate natural effects
    # E[Y(1,M(1))] - observed mean for treated
    E_Y_1_M1 = np.mean(Y_1)
    
    # E[Y(0,M(0))] - observed mean for control
    E_Y_0_M0 = np.mean(Y_0)
    
    # E[Y(1,M(0))] - counterfactual: Y under X=1 but M from X=0
    Y_1_M0 = predict_y(1, M_0.ravel())
    E_Y_1_M0 = np.mean(Y_1_M0)
    
    # E[Y(0,M(1))] - counterfactual: Y under X=0 but M from X=1
    Y_0_M1 = predict_y(0, M_1.ravel())
    E_Y_0_M1 = np.mean(Y_0_M1)
    
    # Natural effects
    total_effect = E_Y_1_M1 - E_Y_0_M0
    nde = E_Y_1_M0 - E_Y_0_M0  # Direct effect
    nie = E_Y_1_M1 - E_Y_1_M0  # Indirect effect
    
    # Alternative decomposition (should sum to total)
    nde_alt = E_Y_1_M1 - E_Y_0_M1
    nie_alt = E_Y_0_M1 - E_Y_0_M0
    
    # Proportion mediated
    prop_mediated = nie / total_effect if abs(total_effect) > 1e-10 else np.nan
    
    results = {
        'method': 'Natural Effects (Binary)',
        'total_effect': total_effect,
        'nde': nde,
        'nie': nie,
        'nde_alt': nde_alt,
        'nie_alt': nie_alt,
        'prop_mediated': prop_mediated,
        'E[Y(1,M(1))]': E_Y_1_M1,
        'E[Y(0,M(0))]': E_Y_0_M0,
        'E[Y(1,M(0))]': E_Y_1_M0,
        'E[Y(0,M(1))]': E_Y_0_M1,
        'interaction_included': interaction,
        'n_treated': len(Y_1),
        'n_control': len(Y_0)
    }
    
    # Bootstrap confidence intervals
    if bootstrap_ci:
        ci_results = _bootstrap_natural_effects_binary(
            X, M, Y, outcome_model.__class__, mediator_model.__class__,
            interaction, n_bootstrap, confidence_level, random_state
        )
        results.update(ci_results)
    
    return results


def _estimate_natural_effects_continuous(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                                        outcome_model: Optional[BaseEstimator] = None,
                                        mediator_model: Optional[BaseEstimator] = None,
                                        interaction: bool = True,
                                        bootstrap_ci: bool = False,
                                        n_bootstrap: int = 1000,
                                        confidence_level: float = 0.95,
                                        random_state: int = 42) -> Dict:
    """
    Natural effects for continuous treatment.
    
    Uses the mediation formula approach for continuous exposures.
    """
    # Default models
    if outcome_model is None:
        outcome_model = LinearRegression()
    if mediator_model is None:
        mediator_model = LinearRegression()
    
    # Ensure proper shapes
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    M = M.reshape(-1, 1) if M.ndim == 1 else M
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    # Fit mediator model
    mediator_model = clone(mediator_model)
    mediator_model.fit(X, M.ravel())
    
    # For continuous X, we estimate effects at specific values
    # or marginal effects
    if interaction:
        # Fit model with interaction
        XM_int = X * M
        features = np.hstack([X, M, XM_int])
        outcome_model = clone(outcome_model)
        outcome_model.fit(features, Y.ravel())
        
        # Get coefficients
        gamma = outcome_model.coef_[0]  # X coefficient
        beta = outcome_model.coef_[1]   # M coefficient  
        theta = outcome_model.coef_[2]  # X*M coefficient
        
        # For linear models with interaction:
        # NDE at M = E[M|X=x0] is γ + θ*E[M|X=x0]
        # NIE depends on the change in M and interaction
        
        # Get mediator model coefficient
        alpha = mediator_model.coef_[0] if hasattr(mediator_model, 'coef_') else None
        
        if alpha is not None:
            # Analytical formulas for linear case
            # Average NDE (averaged over M distribution)
            mean_M = np.mean(M)
            nde = gamma + theta * mean_M
            
            # NIE for unit change in X
            nie = alpha * (beta + theta * np.mean(X))
            
            total_effect = nde + nie
        else:
            # Use simulation for non-linear mediator model
            nde, nie, total_effect = _simulate_continuous_effects(
                X, M, Y, outcome_model, mediator_model, interaction
            )
    else:
        # No interaction - simpler case
        features = np.hstack([X, M])
        outcome_model = clone(outcome_model)
        outcome_model.fit(features, Y.ravel())
        
        gamma = outcome_model.coef_[0]  # Direct effect
        beta = outcome_model.coef_[1]   # M→Y effect
        
        # Get mediator effect
        alpha = mediator_model.coef_[0] if hasattr(mediator_model, 'coef_') else None
        
        if alpha is not None:
            nde = gamma
            nie = alpha * beta
            total_effect = gamma + alpha * beta
        else:
            # Non-linear mediator model
            nde, nie, total_effect = _simulate_continuous_effects(
                X, M, Y, outcome_model, mediator_model, interaction
            )
    
    # Proportion mediated
    prop_mediated = nie / total_effect if abs(total_effect) > 1e-10 else np.nan
    
    results = {
        'method': 'Natural Effects (Continuous)',
        'total_effect': total_effect,
        'nde': nde,
        'nie': nie,
        'prop_mediated': prop_mediated,
        'interaction_included': interaction,
        'note': 'Effects are per unit change in X'
    }
    
    # Add model diagnostics
    if hasattr(outcome_model, 'coef_'):
        results['outcome_coefs'] = outcome_model.coef_
    if hasattr(mediator_model, 'coef_'):
        results['mediator_coef'] = mediator_model.coef_[0]
    
    # Bootstrap confidence intervals
    if bootstrap_ci:
        ci_results = _bootstrap_natural_effects_continuous(
            X, M, Y, outcome_model.__class__, mediator_model.__class__,
            interaction, n_bootstrap, confidence_level, random_state
        )
        results.update(ci_results)
    
    return results


def _simulate_continuous_effects(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                                outcome_model: BaseEstimator,
                                mediator_model: BaseEstimator,
                                interaction: bool,
                                delta: float = None) -> Tuple[float, float, float]:
    """
    Simulate natural effects for continuous treatment using Monte Carlo.
    """
    if delta is None:
        delta = np.std(X) * 0.1  # Small change
    
    n_sim = 1000
    x_base = np.mean(X)
    x_new = x_base + delta
    
    # Simulate M under different X values
    M_base = mediator_model.predict([[x_base]])[0]
    M_new = mediator_model.predict([[x_new]])[0]
    
    # Add noise based on residual variance
    M_residuals = M.ravel() - mediator_model.predict(X)
    sigma_M = np.std(M_residuals)
    
    # Simulate
    np.random.seed(42)
    M_base_sim = np.random.normal(M_base, sigma_M, n_sim)
    M_new_sim = np.random.normal(M_new, sigma_M, n_sim)
    
    # Calculate effects
    if interaction:
        # Y(x_new, M(x_base)) vs Y(x_base, M(x_base))
        features_1 = np.column_stack([
            np.full(n_sim, x_new),
            M_base_sim,
            np.full(n_sim, x_new) * M_base_sim
        ])
        features_0 = np.column_stack([
            np.full(n_sim, x_base),
            M_base_sim,
            np.full(n_sim, x_base) * M_base_sim
        ])
        Y_1_M0 = outcome_model.predict(features_1)
        Y_0_M0 = outcome_model.predict(features_0)
        nde = np.mean(Y_1_M0 - Y_0_M0) / delta
        
        # Y(x_new, M(x_new)) vs Y(x_new, M(x_base))
        features_11 = np.column_stack([
            np.full(n_sim, x_new),
            M_new_sim,
            np.full(n_sim, x_new) * M_new_sim
        ])
        Y_1_M1 = outcome_model.predict(features_11)
        nie = np.mean(Y_1_M1 - Y_1_M0) / delta
    else:
        # Simpler without interaction
        features_1_M0 = np.column_stack([np.full(n_sim, x_new), M_base_sim])
        features_0_M0 = np.column_stack([np.full(n_sim, x_base), M_base_sim])
        features_1_M1 = np.column_stack([np.full(n_sim, x_new), M_new_sim])
        
        Y_1_M0 = outcome_model.predict(features_1_M0)
        Y_0_M0 = outcome_model.predict(features_0_M0)
        Y_1_M1 = outcome_model.predict(features_1_M1)
        
        nde = np.mean(Y_1_M0 - Y_0_M0) / delta
        nie = np.mean(Y_1_M1 - Y_1_M0) / delta
    
    total_effect = nde + nie
    
    return nde, nie, total_effect


def _bootstrap_natural_effects_binary(X, M, Y, outcome_model_class, 
                                    mediator_model_class, interaction,
                                    n_bootstrap, confidence_level, random_state):
    """Bootstrap confidence intervals for binary treatment."""
    n = len(X)
    nde_boot = []
    nie_boot = []
    
    for b in range(n_bootstrap):
        # Resample
        idx = np.random.RandomState(random_state + b).choice(n, n, replace=True)
        X_b = X[idx]
        M_b = M[idx]
        Y_b = Y[idx]
        
        try:
            result = _estimate_natural_effects_binary(
                X_b, M_b, Y_b,
                outcome_model_class(),
                mediator_model_class(),
                interaction,
                bootstrap_ci=False
            )
            nde_boot.append(result['nde'])
            nie_boot.append(result['nie'])
        except:
            continue
    
    # Calculate CIs
    alpha = 1 - confidence_level
    nde_ci = np.percentile(nde_boot, [100*alpha/2, 100*(1-alpha/2)])
    nie_ci = np.percentile(nie_boot, [100*alpha/2, 100*(1-alpha/2)])
    
    return {
        'nde_ci': nde_ci,
        'nie_ci': nie_ci,
        'nde_se': np.std(nde_boot),
        'nie_se': np.std(nie_boot),
        'n_successful_bootstraps': len(nde_boot)
    }


def _bootstrap_natural_effects_continuous(X, M, Y, outcome_model_class,
                                        mediator_model_class, interaction,
                                        n_bootstrap, confidence_level, random_state):
    """Bootstrap confidence intervals for continuous treatment."""
    n = len(X)
    nde_boot = []
    nie_boot = []
    
    for b in range(n_bootstrap):
        # Resample
        idx = np.random.RandomState(random_state + b).choice(n, n, replace=True)
        X_b = X[idx]
        M_b = M[idx]
        Y_b = Y[idx]
        
        try:
            result = _estimate_natural_effects_continuous(
                X_b, M_b, Y_b,
                outcome_model_class(),
                mediator_model_class(),
                interaction,
                bootstrap_ci=False
            )
            nde_boot.append(result['nde'])
            nie_boot.append(result['nie'])
        except:
            continue
    
    # Calculate CIs
    alpha = 1 - confidence_level
    nde_ci = np.percentile(nde_boot, [100*alpha/2, 100*(1-alpha/2)])
    nie_ci = np.percentile(nie_boot, [100*alpha/2, 100*(1-alpha/2)])
    
    return {
        'nde_ci': nde_ci,
        'nie_ci': nie_ci,
        'nde_se': np.std(nde_boot),
        'nie_se': np.std(nie_boot),
        'n_successful_bootstraps': len(nde_boot)
    }


def sensitivity_analysis(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                       rho_values: np.ndarray = None) -> Dict:
    """
    Sensitivity analysis for unmeasured confounding.
    
    Examines how natural effects change under different assumptions
    about unmeasured confounding between M and Y.
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    rho_values : np.ndarray or None
        Correlation values to test (default: -0.5 to 0.5)
        
    Returns
    -------
    dict
        Sensitivity analysis results
    """
    if rho_values is None:
        rho_values = np.linspace(-0.5, 0.5, 11)
    
    results = []
    
    for rho in rho_values:
        # Adjust Y for potential confounding
        # This is a simplified sensitivity analysis
        # In practice, would use more sophisticated methods
        
        # Add correlated error to Y
        n = len(Y)
        U = np.random.normal(0, 1, n)
        Y_adjusted = Y + rho * np.std(Y) * U
        
        # Re-estimate effects
        effects = estimate_natural_effects(X, M, Y_adjusted)
        
        results.append({
            'rho': rho,
            'nde': effects['nde'],
            'nie': effects['nie'],
            'total': effects['total_effect']
        })
    
    return {
        'sensitivity_results': results,
        'rho_values': rho_values,
        'note': 'Simplified sensitivity analysis for illustration'
    }