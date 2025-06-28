"""
Data Generation Module for PoMA Frameworks Comparison

Generates various mediation scenarios to test different frameworks:
1. Linear relationships (baseline)
2. Non-linear relationships
3. Interaction effects
4. Edge cases (symmetric, threshold, etc.)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class MediationData:
    """Container for mediation analysis data"""
    X: np.ndarray  # Treatment/exposure
    M: np.ndarray  # Mediator
    Y: np.ndarray  # Outcome
    scenario: str  # Description
    true_effects: Dict[str, float]  # True causal effects if known
    
    @property
    def n(self) -> int:
        return len(self.X)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {'X': self.X, 'M': self.M, 'Y': self.Y}


class DataGenerator:
    """Generate various mediation scenarios"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    # ========== Linear Scenarios ==========
    
    def linear_mediation(self, n: int = 1000, 
                        alpha: float = 1.5,  # X→M effect
                        beta: float = 2.0,   # M→Y effect
                        gamma: float = 0.5,  # X→Y direct effect
                        noise_scale: float = 0.5) -> MediationData:
        """
        Classic linear mediation model:
        M = α*X + ε_M
        Y = γ*X + β*M + ε_Y
        """
        X = np.random.normal(0, 1, n)
        M = alpha * X + np.random.normal(0, noise_scale, n)
        Y = gamma * X + beta * M + np.random.normal(0, noise_scale, n)
        
        # True effects
        total_effect = gamma + alpha * beta
        direct_effect = gamma
        indirect_effect = alpha * beta
        poma_true = indirect_effect / total_effect if total_effect != 0 else 0
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Linear Mediation",
            true_effects={
                'total': total_effect,
                'direct': direct_effect,
                'indirect': indirect_effect,
                'poma': poma_true,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma
            }
        )
    
    def binary_treatment_linear(self, n: int = 1000,
                               p_treat: float = 0.5,
                               alpha: float = 2.0,
                               beta: float = 1.5,
                               gamma: float = 1.0) -> MediationData:
        """Linear mediation with binary treatment"""
        X = np.random.binomial(1, p_treat, n)
        M = alpha * X + np.random.normal(0, 0.5, n)
        Y = gamma * X + beta * M + np.random.normal(0, 0.5, n)
        
        total_effect = gamma + alpha * beta
        direct_effect = gamma
        indirect_effect = alpha * beta
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Binary Treatment Linear",
            true_effects={
                'total': total_effect,
                'direct': direct_effect,
                'indirect': indirect_effect,
                'poma': indirect_effect / total_effect
            }
        )
    
    # ========== Non-linear Scenarios ==========
    
    def polynomial_mediator(self, n: int = 1000,
                           degree: int = 2) -> MediationData:
        """
        Non-linear M→Y relationship:
        M = 1.5*X + ε
        Y = 0.5*X + M - 0.2*M² + ε
        """
        X = np.random.normal(0, 1, n)
        M = 1.5 * X + np.random.normal(0, 0.5, n)
        Y = 0.5 * X + M - 0.2 * M**2 + np.random.normal(0, 0.5, n)
        
        # Approximate effects (linearized around mean)
        mean_M = 0
        beta_approx = 1 - 0.4 * mean_M
        total_approx = 0.5 + 1.5 * beta_approx
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario=f"Polynomial Mediator (degree {degree})",
            true_effects={
                'total_approx': total_approx,
                'direct': 0.5,
                'note': 'Non-linear relationship makes exact decomposition complex'
            }
        )
    
    def exponential_mediator(self, n: int = 1000) -> MediationData:
        """Exponential M→Y relationship"""
        X = np.random.uniform(0, 2, n)
        M = 2 * X + np.random.normal(0, 0.3, n)
        Y = 0.3 * X + 2 * (1 - np.exp(-M/2)) + np.random.normal(0, 0.3, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Exponential Mediator",
            true_effects={'note': 'Non-linear, no simple decomposition'}
        )
    
    def threshold_effect(self, n: int = 1000,
                        threshold: float = 0) -> MediationData:
        """Threshold effect in M→Y"""
        X = np.random.normal(0, 1, n)
        M = 1.5 * X + np.random.normal(0, 0.5, n)
        Y = 0.5 * X + 2 * (M > threshold) + np.random.normal(0, 0.5, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario=f"Threshold Effect (M > {threshold})",
            true_effects={'threshold': threshold}
        )
    
    # ========== Interaction Scenarios ==========
    
    def linear_interaction(self, n: int = 1000,
                          alpha: float = 2.0,
                          beta: float = 1.5,
                          gamma: float = 0.5,
                          delta: float = 0.8) -> MediationData:
        """
        Treatment-mediator interaction:
        M = α*X + ε
        Y = γ*X + β*M + δ*X*M + ε
        """
        X = np.random.normal(0, 1, n)
        M = alpha * X + np.random.normal(0, 0.5, n)
        Y = gamma * X + beta * M + delta * X * M + np.random.normal(0, 0.5, n)
        
        # With interaction, effects depend on levels
        # At X=0: effect through M is β
        # At X=1: effect through M is β + δ
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Linear with Interaction",
            true_effects={
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'delta': delta,
                'note': 'Effects vary with treatment level'
            }
        )
    
    def binary_interaction(self, n: int = 1000) -> MediationData:
        """Binary treatment with interaction"""
        X = np.random.binomial(1, 0.5, n)
        M = 2 * X + 0.5 * X + np.random.normal(0, 0.5, n)
        Y = 1 + 0.5 * X + 2 * M + 0.5 * X * M + np.random.normal(0, 0.5, n)
        
        # Natural effects can be calculated analytically here
        # NDE = E[Y(1,M(0))] - E[Y(0,M(0))]
        # NIE = E[Y(1,M(1))] - E[Y(1,M(0))]
        
        # E[M|X=0] ≈ 0, E[M|X=1] ≈ 2.5
        NDE = 0.5  # Direct effect when M at X=0 level
        NIE = 2.5 * 2.5  # Effect through change in M (including interaction)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Binary Treatment with Interaction",
            true_effects={
                'NDE': NDE,
                'NIE': NIE,
                'total': NDE + NIE
            }
        )
    
    # ========== Edge Cases ==========
    
    def symmetric_ushaped(self, n: int = 1000) -> MediationData:
        """
        Symmetric U-shaped relationships:
        X ~ Uniform(-3, 3)
        M = X²
        Y = (M - 5)²
        """
        X = np.random.uniform(-3, 3, n)
        M = X**2 + np.random.normal(0, 0.3, n)
        Y = (M - 5)**2 + np.random.normal(0, 0.5, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Symmetric U-shaped",
            true_effects={
                'note': 'Zero correlation, traditional methods fail',
                'corr_XY': np.corrcoef(X, Y)[0, 1]
            }
        )
    
    def perfect_mediation(self, n: int = 1000) -> MediationData:
        """Complete mediation with no direct effect"""
        X = np.random.normal(0, 1, n)
        M = 2 * X + np.random.normal(0, 0.3, n)
        Y = 3 * M + np.random.normal(0, 0.3, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Perfect Mediation",
            true_effects={
                'total': 6.0,
                'direct': 0.0,
                'indirect': 6.0,
                'poma': 1.0
            }
        )
    
    def suppression_effect(self, n: int = 1000) -> MediationData:
        """Suppressor variable scenario"""
        # X contains signal + noise
        signal = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)
        X = signal + noise
        
        # M captures the noise
        M = noise + np.random.normal(0, 0.3, n)
        
        # Y depends on signal only
        Y = 2 * signal + np.random.normal(0, 0.5, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Suppression Effect",
            true_effects={
                'note': 'Controlling for M increases X→Y effect'
            }
        )
    
    def competitive_mediation(self, n: int = 1000) -> MediationData:
        """Two competing pathways"""
        X = np.random.normal(0, 1, n)
        
        # Two mediators (we'll combine into one for simplicity)
        M1 = 2 * X + np.random.normal(0, 0.3, n)  # Positive pathway
        M2 = -1.5 * X + np.random.normal(0, 0.3, n)  # Negative pathway
        
        # Combined mediator (for 3-variable analysis)
        M = M1 + M2  # Net effect
        
        # Y affected by both pathways
        Y = 0.5 * X + 1.5 * M1 + 1.0 * M2 + np.random.normal(0, 0.5, n)
        
        return MediationData(
            X=X, M=M, Y=Y,
            scenario="Competitive Mediation",
            true_effects={
                'note': 'Opposing pathways can lead to strange PoMA values'
            }
        )
    
    # ========== Utility Methods ==========
    
    def generate_scenario(self, scenario_name: str, n: int = 1000, **kwargs) -> MediationData:
        """Generate data for a named scenario"""
        scenarios = {
            'linear': self.linear_mediation,
            'binary': self.binary_treatment_linear,
            'polynomial': self.polynomial_mediator,
            'exponential': self.exponential_mediator,
            'threshold': self.threshold_effect,
            'interaction': self.linear_interaction,
            'binary_interaction': self.binary_interaction,
            'symmetric': self.symmetric_ushaped,
            'perfect': self.perfect_mediation,
            'suppression': self.suppression_effect,
            'competitive': self.competitive_mediation
        }
        
        if scenario_name not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                           f"Available: {list(scenarios.keys())}")
        
        return scenarios[scenario_name](n=n, **kwargs)
    
    def generate_all_scenarios(self, n: int = 1000) -> Dict[str, MediationData]:
        """Generate all available scenarios"""
        scenarios = {}
        for name in ['linear', 'binary', 'polynomial', 'exponential', 
                    'threshold', 'interaction', 'binary_interaction',
                    'symmetric', 'perfect', 'suppression', 'competitive']:
            scenarios[name] = self.generate_scenario(name, n=n)
        return scenarios


# Convenience functions
def generate_linear_data(n: int = 1000, **kwargs) -> MediationData:
    """Quick generation of linear mediation data"""
    gen = DataGenerator()
    return gen.linear_mediation(n=n, **kwargs)


def generate_interaction_data(n: int = 1000, **kwargs) -> MediationData:
    """Quick generation of interaction data"""
    gen = DataGenerator()
    return gen.linear_interaction(n=n, **kwargs)