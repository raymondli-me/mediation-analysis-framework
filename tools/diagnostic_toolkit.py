#!/usr/bin/env python3
"""
Mediation Analysis Diagnostic Toolkit
====================================

This toolkit helps practitioners determine which mediation analysis method
is most appropriate for their data. It includes:

1. Data diagnostics (linearity, interactions, distributions)
2. Method recommendations based on data characteristics
3. Assumption checking for each framework
4. Visualization tools for mediation relationships

Usage:
    from diagnostic_toolkit import diagnose_mediation_data
    report = diagnose_mediation_data(X, M, Y)
    report.show()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy import stats
from typing import Dict, Tuple, List, Optional
import warnings


class MediationDiagnosticReport:
    """
    Container for diagnostic results with visualization methods.
    """
    
    def __init__(self, diagnostics: Dict):
        self.diagnostics = diagnostics
        self.recommendations = []
        self.warnings = []
        self._generate_recommendations()
    
    def _generate_recommendations(self):
        """Generate method recommendations based on diagnostics."""
        d = self.diagnostics
        
        # Check for near-zero correlations
        if abs(d['correlations']['X-Y']) < 0.05:
            self.warnings.append("⚠️ Very low X-Y correlation. PoMA may be unstable.")
        
        # Check treatment type
        if d['treatment_type'] == 'binary':
            self.recommendations.append("✓ Binary treatment - all methods applicable")
        else:
            self.recommendations.append("📊 Continuous treatment - check linearity carefully")
        
        # Check for non-linearity
        if d['nonlinearity']['X-M']['significant'] or d['nonlinearity']['M-Y']['significant']:
            self.recommendations.append("📈 Non-linear relationships detected - use DML with ML")
            if d['nonlinearity']['M-Y']['r2_improvement'] > 0.1:
                self.recommendations.append("   → Strong M→Y non-linearity - ML models essential")
        else:
            self.recommendations.append("✓ Linear relationships - traditional methods sufficient")
        
        # Check for interaction
        if d['interaction']['significant']:
            self.recommendations.append("🔄 Interaction detected - use Natural Effects")
            self.recommendations.append(f"   → Interaction strength: {d['interaction']['strength']}")
        
        # Check for multicollinearity
        if d['multicollinearity']['vif'] > 5:
            self.warnings.append(f"⚠️ High multicollinearity (VIF={d['multicollinearity']['vif']:.1f})")
        
        # Check sample size
        if d['sample_size'] < 200:
            self.warnings.append("⚠️ Small sample size - avoid complex ML models")
        elif d['sample_size'] < 500:
            self.recommendations.append("📊 Moderate sample size - simple ML models ok")
        else:
            self.recommendations.append("✓ Large sample size - all methods feasible")
        
        # Final method ranking
        self._rank_methods()
    
    def _rank_methods(self):
        """Rank methods based on data characteristics."""
        d = self.diagnostics
        
        methods = {
            'Traditional': 100,  # Base score
            'FWL': 100,
            'DML-Linear': 95,
            'DML-ML': 50,
            'Natural-Linear': 80,
            'Natural-ML': 40
        }
        
        # Adjust scores based on diagnostics
        if d['nonlinearity']['X-M']['significant'] or d['nonlinearity']['M-Y']['significant']:
            methods['Traditional'] -= 30
            methods['FWL'] -= 30
            methods['DML-Linear'] -= 20
            methods['DML-ML'] += 40
            methods['Natural-Linear'] -= 20
            methods['Natural-ML'] += 30
        
        if d['interaction']['significant']:
            methods['Traditional'] -= 20
            methods['FWL'] -= 20
            methods['DML-Linear'] -= 20
            methods['DML-ML'] -= 20
            methods['Natural-Linear'] += 30
            methods['Natural-ML'] += 40
        
        if d['sample_size'] < 500:
            methods['DML-ML'] -= 20
            methods['Natural-ML'] -= 20
        
        # Sort and store
        self.method_ranking = sorted(methods.items(), key=lambda x: x[1], reverse=True)
    
    def show(self, save_path: Optional[str] = None):
        """Display comprehensive diagnostic report."""
        print("="*80)
        print("MEDIATION ANALYSIS DIAGNOSTIC REPORT")
        print("="*80)
        
        # Basic info
        print(f"\nSample size: {self.diagnostics['sample_size']}")
        print(f"Treatment type: {self.diagnostics['treatment_type']}")
        
        # Correlations
        print("\nCorrelations:")
        for pair, corr in self.diagnostics['correlations'].items():
            print(f"  {pair}: {corr:.3f}")
        
        # Warnings
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Recommendations
        print("\nRecommendations:")
        for rec in self.recommendations:
            print(f"  {rec}")
        
        # Method ranking
        print("\nRecommended methods (in order):")
        for i, (method, score) in enumerate(self.method_ranking[:3]):
            print(f"  {i+1}. {method} (score: {score})")
        
        # Create visualizations
        self.create_diagnostic_plots(save_path)
    
    def create_diagnostic_plots(self, save_path: Optional[str] = None):
        """Create diagnostic visualizations."""
        fig = plt.figure(figsize=(16, 12))
        
        # Get data from diagnostics
        X = self.diagnostics['_data']['X']
        M = self.diagnostics['_data']['M']
        Y = self.diagnostics['_data']['Y']
        
        # 1. Scatter plots
        ax1 = plt.subplot(3, 3, 1)
        self._plot_relationship(X, M, ax1, "X → M")
        
        ax2 = plt.subplot(3, 3, 2)
        self._plot_relationship(M, Y, ax2, "M → Y")
        
        ax3 = plt.subplot(3, 3, 3)
        self._plot_relationship(X, Y, ax3, "X → Y")
        
        # 2. Residual plots
        ax4 = plt.subplot(3, 3, 4)
        self._plot_residuals(X, M, ax4, "X → M Residuals")
        
        ax5 = plt.subplot(3, 3, 5)
        self._plot_residuals(M, Y, ax5, "M → Y Residuals")
        
        ax6 = plt.subplot(3, 3, 6)
        self._plot_interaction(X, M, Y, ax6)
        
        # 3. Distribution plots
        ax7 = plt.subplot(3, 3, 7)
        self._plot_distribution(X, ax7, "X Distribution")
        
        ax8 = plt.subplot(3, 3, 8)
        self._plot_distribution(M, ax8, "M Distribution")
        
        ax9 = plt.subplot(3, 3, 9)
        self._plot_distribution(Y, ax9, "Y Distribution")
        
        plt.suptitle("Mediation Analysis Diagnostics", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_relationship(self, x, y, ax, title):
        """Plot relationship with linear and polynomial fits."""
        ax.scatter(x, y, alpha=0.5, s=10)
        
        # Linear fit
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_linear = lr.predict(x_range)
        ax.plot(x_range, y_linear, 'b-', label='Linear', linewidth=2)
        
        # Polynomial fit
        poly = PolynomialFeatures(degree=2)
        x_poly = poly.fit_transform(x.reshape(-1, 1))
        lr_poly = LinearRegression()
        lr_poly.fit(x_poly, y)
        x_range_poly = poly.transform(x_range)
        y_poly = lr_poly.predict(x_range_poly)
        ax.plot(x_range, y_poly, 'r--', label='Quadratic', linewidth=2)
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals(self, x, y, ax, title):
        """Plot residual patterns."""
        lr = LinearRegression()
        lr.fit(x.reshape(-1, 1), y)
        y_pred = lr.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add LOESS smooth if significant pattern
        from scipy.signal import savgol_filter
        sorted_idx = np.argsort(y_pred)
        window = min(51, len(y_pred) // 4)
        if window % 2 == 0:
            window += 1
        if window >= 5:
            smooth = savgol_filter(residuals[sorted_idx], window, 3)
            ax.plot(y_pred[sorted_idx], smooth, 'g-', linewidth=2, alpha=0.7)
    
    def _plot_interaction(self, x, m, y, ax):
        """Plot interaction effect."""
        # For binary X, show different M-Y slopes
        if self.diagnostics['treatment_type'] == 'binary':
            x_unique = np.unique(x)
            colors = ['blue', 'red']
            for i, x_val in enumerate(x_unique):
                mask = x == x_val
                ax.scatter(m[mask], y[mask], alpha=0.5, s=10, 
                          color=colors[i], label=f'X={int(x_val)}')
                
                # Fit line
                if sum(mask) > 10:
                    lr = LinearRegression()
                    lr.fit(m[mask].reshape(-1, 1), y[mask])
                    m_range = np.linspace(m[mask].min(), m[mask].max(), 100)
                    y_pred = lr.predict(m_range.reshape(-1, 1))
                    ax.plot(m_range, y_pred, color=colors[i], linewidth=2)
        else:
            # For continuous X, use color gradient
            scatter = ax.scatter(m, y, c=x, cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='X value')
        
        ax.set_xlabel('M')
        ax.set_ylabel('Y')
        ax.set_title('Interaction Check (M→Y by X)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distribution(self, data, ax, title):
        """Plot distribution with normality test."""
        ax.hist(data, bins=30, density=True, alpha=0.7, color='blue')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add normal fit
        mu, sigma = stats.norm.fit(data)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 
               'g--', linewidth=2, label='Normal fit')
        
        # Normality test
        stat, p_value = stats.shapiro(data[:min(5000, len(data))])
        ax.text(0.05, 0.95, f'Shapiro p={p_value:.3f}', 
               transform=ax.transAxes, verticalalignment='top')
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)


def diagnose_mediation_data(X: np.ndarray, M: np.ndarray, Y: np.ndarray) -> MediationDiagnosticReport:
    """
    Comprehensive diagnostic analysis for mediation data.
    
    Parameters
    ----------
    X : np.ndarray
        Treatment/exposure variable
    M : np.ndarray
        Mediator variable
    Y : np.ndarray
        Outcome variable
    
    Returns
    -------
    MediationDiagnosticReport
        Diagnostic report with recommendations
    """
    diagnostics = {}
    
    # Ensure proper shapes
    X = X.ravel()
    M = M.ravel()
    Y = Y.ravel()
    n = len(X)
    
    # Store data for plotting
    diagnostics['_data'] = {'X': X, 'M': M, 'Y': Y}
    diagnostics['sample_size'] = n
    
    # 1. Treatment type
    X_unique = np.unique(X)
    is_binary = len(X_unique) == 2 and set(X_unique).issubset({0, 1})
    diagnostics['treatment_type'] = 'binary' if is_binary else 'continuous'
    
    # 2. Basic correlations
    diagnostics['correlations'] = {
        'X-Y': np.corrcoef(X, Y)[0, 1],
        'X-M': np.corrcoef(X, M)[0, 1],
        'M-Y': np.corrcoef(M, Y)[0, 1]
    }
    
    # 3. Check for non-linearity
    diagnostics['nonlinearity'] = {
        'X-M': _check_nonlinearity(X, M),
        'M-Y': _check_nonlinearity(M, Y)
    }
    
    # 4. Check for interaction
    diagnostics['interaction'] = _check_interaction(X, M, Y)
    
    # 5. Check distributions
    diagnostics['distributions'] = {
        'X': _check_distribution(X),
        'M': _check_distribution(M),
        'Y': _check_distribution(Y)
    }
    
    # 6. Multicollinearity check
    diagnostics['multicollinearity'] = _check_multicollinearity(X, M)
    
    # 7. Outlier detection
    diagnostics['outliers'] = _detect_outliers(X, M, Y)
    
    # 8. Effect size estimates (preliminary)
    diagnostics['preliminary_effects'] = _estimate_preliminary_effects(X, M, Y)
    
    return MediationDiagnosticReport(diagnostics)


def _check_nonlinearity(x: np.ndarray, y: np.ndarray) -> Dict:
    """Check for non-linear relationship."""
    # Linear model
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    r2_linear = lr.score(x.reshape(-1, 1), y)
    
    # Polynomial model
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    lr_poly = LinearRegression()
    lr_poly.fit(x_poly, y)
    r2_poly = lr_poly.score(x_poly, y)
    
    # F-test for improvement
    n = len(x)
    k1 = 2  # Linear model parameters
    k2 = 4  # Polynomial model parameters
    
    if r2_linear < 0.999:  # Avoid division issues
        f_stat = ((r2_poly - r2_linear) / (k2 - k1)) / ((1 - r2_poly) / (n - k2))
        p_value = 1 - stats.f.cdf(f_stat, k2 - k1, n - k2)
    else:
        f_stat = 0
        p_value = 1
    
    return {
        'r2_linear': r2_linear,
        'r2_poly': r2_poly,
        'r2_improvement': r2_poly - r2_linear,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05 and (r2_poly - r2_linear) > 0.02
    }


def _check_interaction(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> Dict:
    """Check for treatment-mediator interaction."""
    # Model without interaction
    X_no_int = np.column_stack([x, m])
    lr_no_int = LinearRegression()
    lr_no_int.fit(X_no_int, y)
    r2_no_int = lr_no_int.score(X_no_int, y)
    
    # Model with interaction
    X_int = np.column_stack([x, m, x * m])
    lr_int = LinearRegression()
    lr_int.fit(X_int, y)
    r2_int = lr_int.score(X_int, y)
    interaction_coef = lr_int.coef_[2]
    
    # Calculate standard error (approximate)
    n = len(y)
    residuals = y - lr_int.predict(X_int)
    mse = np.mean(residuals**2)
    
    # Approximate SE using simple formula
    X_int_centered = X_int - np.mean(X_int, axis=0)
    var_interaction = np.sum(X_int_centered[:, 2]**2)
    se_interaction = np.sqrt(mse / var_interaction) if var_interaction > 0 else np.inf
    
    # T-test
    t_stat = interaction_coef / se_interaction if se_interaction < np.inf else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 4))
    
    # Effect size
    if np.std(x) > 0 and np.std(m) > 0:
        interaction_std = interaction_coef * np.std(x) * np.std(m) / np.std(y)
    else:
        interaction_std = 0
    
    return {
        'coefficient': interaction_coef,
        'se': se_interaction,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'r2_improvement': r2_int - r2_no_int,
        'strength': 'strong' if abs(interaction_std) > 0.2 else 'moderate' if abs(interaction_std) > 0.1 else 'weak'
    }


def _check_distribution(data: np.ndarray) -> Dict:
    """Check distribution properties."""
    # Normality tests
    if len(data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        # Use subsample for large data
        subsample = np.random.choice(data, 5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(subsample)
    
    # Skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'shapiro_p': shapiro_p,
        'normal': shapiro_p > 0.05 and abs(skewness) < 1 and abs(kurtosis) < 3
    }


def _check_multicollinearity(x: np.ndarray, m: np.ndarray) -> Dict:
    """Check multicollinearity between X and M."""
    # Simple correlation
    corr = np.corrcoef(x, m)[0, 1]
    
    # VIF approximation
    r2 = corr**2
    vif = 1 / (1 - r2) if r2 < 0.999 else np.inf
    
    return {
        'correlation': corr,
        'r2': r2,
        'vif': vif,
        'high': vif > 5
    }


def _detect_outliers(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> Dict:
    """Detect outliers using multiple methods."""
    def iqr_outliers(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return np.sum((data < lower) | (data > upper))
    
    # Mahalanobis distance for multivariate outliers
    data_matrix = np.column_stack([x, m, y])
    mean = np.mean(data_matrix, axis=0)
    cov_matrix = np.cov(data_matrix.T)
    
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        mahal_dist = np.array([
            np.sqrt((row - mean).T @ inv_cov @ (row - mean))
            for row in data_matrix
        ])
        # Chi-square threshold
        threshold = np.sqrt(stats.chi2.ppf(0.975, 3))
        multivariate_outliers = np.sum(mahal_dist > threshold)
    except np.linalg.LinAlgError:
        multivariate_outliers = 0
    
    return {
        'X_outliers': iqr_outliers(x),
        'M_outliers': iqr_outliers(m),
        'Y_outliers': iqr_outliers(y),
        'multivariate_outliers': multivariate_outliers,
        'total_outliers': multivariate_outliers,
        'percentage': 100 * multivariate_outliers / len(x)
    }


def _estimate_preliminary_effects(x: np.ndarray, m: np.ndarray, y: np.ndarray) -> Dict:
    """Quick preliminary effect estimates."""
    # Total effect
    lr_total = LinearRegression()
    lr_total.fit(x.reshape(-1, 1), y)
    total_effect = lr_total.coef_[0]
    
    # Direct effect (controlling for M)
    X_with_m = np.column_stack([x, m])
    lr_direct = LinearRegression()
    lr_direct.fit(X_with_m, y)
    direct_effect = lr_direct.coef_[0]
    
    # Indirect effect
    indirect_effect = total_effect - direct_effect
    
    # Proportion mediated
    if abs(total_effect) > 1e-10:
        prop_mediated = indirect_effect / total_effect
    else:
        prop_mediated = np.nan
    
    return {
        'total': total_effect,
        'direct': direct_effect,
        'indirect': indirect_effect,
        'prop_mediated': prop_mediated,
        'method': 'Linear regression (preliminary)'
    }


def plot_method_comparison(X: np.ndarray, M: np.ndarray, Y: np.ndarray,
                          methods_results: Dict[str, Dict]) -> plt.Figure:
    """
    Create a visual comparison of different methods' results.
    
    Parameters
    ----------
    X, M, Y : np.ndarray
        Data arrays
    methods_results : dict
        Results from different methods
    
    Returns
    -------
    plt.Figure
        Comparison figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    methods = list(methods_results.keys())
    
    # 1. Direct effects
    ax = axes[0, 0]
    direct_effects = [methods_results[m].get('direct_effect', np.nan) for m in methods]
    ax.bar(methods, direct_effects, alpha=0.7)
    ax.set_ylabel('Direct Effect')
    ax.set_title('Direct Effects by Method')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 2. Indirect effects
    ax = axes[0, 1]
    indirect_effects = [methods_results[m].get('indirect_effect', np.nan) for m in methods]
    ax.bar(methods, indirect_effects, alpha=0.7, color='orange')
    ax.set_ylabel('Indirect Effect')
    ax.set_title('Indirect Effects by Method')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # 3. PoMA
    ax = axes[1, 0]
    pomas = [methods_results[m].get('poma', np.nan) for m in methods]
    ax.bar(methods, pomas, alpha=0.7, color='green')
    ax.set_ylabel('PoMA')
    ax.set_title('Proportion Mediated by Method')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence intervals (if available)
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        if 'ci_direct' in methods_results[method]:
            ci = methods_results[method]['ci_direct']
            de = methods_results[method]['direct_effect']
            ax.errorbar(i, de, yerr=[[de - ci[0]], [ci[1] - de]], 
                       fmt='o', capsize=5, label=method)
    
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylabel('Direct Effect')
    ax.set_title('Direct Effects with Confidence Intervals')
    ax.grid(True, alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    
    plt.suptitle('Method Comparison Results', fontsize=14)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("Mediation Diagnostic Toolkit")
    print("="*80)
    print("\nExample usage:")
    print("  from diagnostic_toolkit import diagnose_mediation_data")
    print("  report = diagnose_mediation_data(X, M, Y)")
    print("  report.show()")
    print("\nGenerating example with interaction...")
    
    # Generate example data with interaction
    np.random.seed(42)
    n = 1000
    X = np.random.binomial(1, 0.5, n)
    M = 2 * X + np.random.normal(0, 0.5, n)
    Y = 1 + 0.5 * X + 2 * M + 0.8 * X * M + np.random.normal(0, 0.5, n)
    
    # Run diagnostics
    report = diagnose_mediation_data(X, M, Y)
    report.show()