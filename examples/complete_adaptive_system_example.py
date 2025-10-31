"""
Complete Adaptive System Example

This example demonstrates the full integration of ChannelPy's adaptive components:
- Feature scoring for multi-dimensional evaluation
- Topology-aware adaptive thresholds
- Multi-scale regime detection
- Real-world applications with intelligent decision making

This showcases the complete ChannelPy adaptive ecosystem in action.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
import time

# Import all adaptive components
from channelpy.adaptive import (
    FeatureScorer, 
    TopologyAdaptiveThreshold,
    MultiScaleAdaptiveThreshold,
    create_trading_scorer,
    create_medical_scorer,
    create_signal_scorer,
    RegimeType, RegimeChange
)
from channelpy import StateArray, EMPTY, DELTA, PHI, PSI


def generate_complex_streaming_data(n_samples: int = 3000) -> np.ndarray:
    """Generate complex streaming data with multiple regime changes"""
    np.random.seed(42)
    
    data = []
    
    # Phase 1: Stable regime (samples 0-600)
    data.extend(np.random.normal(0, 0.5, 600))
    
    # Phase 2: Trending regime (samples 600-1200)
    trend = np.linspace(0, 3, 600)
    noise = np.random.normal(0, 0.3, 600)
    data.extend(trend + noise)
    
    # Phase 3: Volatile regime (samples 1200-1800)
    data.extend(np.random.normal(0, 2.0, 600))
    
    # Phase 4: Mean reverting regime (samples 1800-2400)
    mean_reverting = [0]
    for i in range(599):
        prev = mean_reverting[-1]
        new_val = 0.8 * prev + np.random.normal(0, 0.4)
        mean_reverting.append(new_val)
    data.extend(mean_reverting)
    
    # Phase 5: Transitioning regime (samples 2400-3000)
    transition = np.linspace(0, -1, 600)
    noise = np.random.normal(0, 0.6, 600)
    data.extend(transition + noise)
    
    return np.array(data)


def create_decision_interpreter() -> Callable:
    """Create a decision interpreter for channel states"""
    
    def interpret(state) -> Dict[str, Any]:
        """
        Interpret channel state into actionable decision
        
        Parameters
        ----------
        state : State
            Channel state to interpret
            
        Returns
        -------
        decision : Dict
            Decision with action, confidence, and reasoning
        """
        if state == PSI:
            return {
                'action': 'STRONG_BUY',
                'confidence': 0.95,
                'reasoning': 'High presence and membership - strong signal',
                'risk_level': 'HIGH'
            }
        elif state == PHI:
            return {
                'action': 'BUY',
                'confidence': 0.75,
                'reasoning': 'High presence, low membership - moderate signal',
                'risk_level': 'MEDIUM'
            }
        elif state == DELTA:
            return {
                'action': 'HOLD',
                'confidence': 0.60,
                'reasoning': 'Low presence, high membership - weak signal',
                'risk_level': 'LOW'
            }
        else:  # EMPTY
            return {
                'action': 'SELL',
                'confidence': 0.80,
                'reasoning': 'Low presence and membership - negative signal',
                'risk_level': 'MEDIUM'
            }
    
    return interpret


def demonstrate_basic_integration():
    """Demonstrate basic integration of adaptive components"""
    print("=== Basic Adaptive System Integration ===")
    
    # Generate data
    data = generate_complex_streaming_data(2000)
    
    # Create scorer
    scorer = create_trading_scorer()
    
    # Create topology-aware tracker
    topology_threshold = TopologyAdaptiveThreshold(
        window_size=1000,
        adaptation_rate=0.01,
        topology_update_interval=100,
        feature_scorer=scorer
    )
    
    # Create decision interpreter
    interpret = create_decision_interpreter()
    
    # Process data
    states = []
    decisions = []
    scores = []
    
    print("Processing data with integrated adaptive system...")
    
    for i, value in enumerate(data):
        # Update topology-aware threshold
        topology_threshold.update(value)
        state = topology_threshold.encode(value)
        states.append(state)
        
        # Score the feature if we have enough history
        if i > 100:
            context = {
                'historical_values': data[max(0, i-100):i].tolist(),
                'historical_outcomes': np.diff(data[max(0, i-100):i+1]).tolist(),
                'sample_size': min(100, i),
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.1
            }
            
            score, dim_scores = scorer.score_and_aggregate(value, context)
            scores.append(score)
            
            # Record score for analysis
            scorer.record_score(value, context, outcome=value)
        else:
            scores.append(0.5)
        
        # Make decision
        decision = interpret(state)
        decisions.append(decision)
        
        # Print progress
        if (i + 1) % 500 == 0:
            print(f"  Sample {i+1}: state={state}, action={decision['action']}, "
                  f"confidence={decision['confidence']:.3f}")
    
    # Analyze results
    states_array = StateArray(states)
    
    print(f"\nBasic Integration Results:")
    print(f"  Total samples: {len(data)}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Average score: {np.mean(scores):.3f}")
    
    # Decision analysis
    actions = [d['action'] for d in decisions]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"  Decision distribution: {action_counts}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot 1: Data
    axes[0].plot(data, alpha=0.7, label='Data')
    axes[0].set_title('Streaming Data with Adaptive Processing')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: States
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Channel States')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[2].plot(scores, 'g-', linewidth=2, alpha=0.8)
    axes[2].set_title('Feature Scores')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Decisions
    decision_map = {'STRONG_BUY': 3, 'BUY': 2, 'HOLD': 1, 'SELL': 0}
    decision_ints = [decision_map.get(d['action'], 0) for d in decisions]
    axes[3].plot(decision_ints, 'purple', linewidth=2, alpha=0.8)
    axes[3].set_title('Decisions')
    axes[3].set_xlabel('Sample')
    axes[3].set_ylabel('Decision')
    axes[3].set_ylim(-0.5, 3.5)
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(['SELL', 'HOLD', 'BUY', 'STRONG_BUY'])
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_adaptive_basic.png', dpi=150, bbox_inches='tight')
    print("Saved: complete_adaptive_basic.png")
    
    return states_array, decisions, scores, scorer


def demonstrate_multiscale_integration():
    """Demonstrate multi-scale adaptive system integration"""
    print("\n=== Multi-Scale Adaptive System Integration ===")
    
    # Generate data
    data = generate_complex_streaming_data(3000)
    
    # Create scorer
    scorer = create_trading_scorer()
    
    # Create multi-scale tracker
    multiscale = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=200,
        medium_window=800,
        slow_window=2000
    )
    
    # Create decision interpreter
    interpret = create_decision_interpreter()
    
    # Process data
    states = []
    decisions = []
    regime_changes = []
    scores = []
    
    print("Processing data with multi-scale adaptive system...")
    
    for i, value in enumerate(data):
        # Update multi-scale tracker
        multiscale.update(value)
        
        # Check for regime changes
        if multiscale.regime_changed():
            change = multiscale.get_last_regime_change()
            regime_changes.append(change)
            print(f"  Regime change at sample {i}: {change.from_regime.value} → {change.to_regime.value} "
                  f"(confidence: {change.confidence:.3f})")
        
        # Encode with regime-appropriate threshold
        state = multiscale.encode_adaptive(value)
        states.append(state)
        
        # Score the feature
        if i > 100:
            context = {
                'historical_values': data[max(0, i-100):i].tolist(),
                'historical_outcomes': np.diff(data[max(0, i-100):i+1]).tolist(),
                'sample_size': min(100, i),
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.1
            }
            
            score, dim_scores = scorer.score_and_aggregate(value, context)
            scores.append(score)
            
            # Record score
            scorer.record_score(value, context, outcome=value)
        else:
            scores.append(0.5)
        
        # Make decision based on state
        decision = interpret(state)
        decisions.append(decision)
        
        # Print progress
        if (i + 1) % 750 == 0:
            regime_info = multiscale.get_regime_info()
            print(f"  Sample {i+1}: regime={regime_info['current_regime']}, "
                  f"state={state}, action={decision['action']}")
    
    # Analyze results
    states_array = StateArray(states)
    
    print(f"\nMulti-Scale Integration Results:")
    print(f"  Total samples: {len(data)}")
    print(f"  Regime changes: {len(regime_changes)}")
    print(f"  Final regime: {multiscale.get_current_regime().value}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Average score: {np.mean(scores):.3f}")
    
    # Regime analysis
    regime_counts = {}
    for change in regime_changes:
        regime = change.to_regime.value
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    
    print(f"  Regime distribution: {regime_counts}")
    
    # Plot results
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    
    # Plot 1: Data with regime changes
    axes[0].plot(data, alpha=0.7, label='Data')
    
    # Mark regime changes
    for change in regime_changes:
        axes[0].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
    
    axes[0].set_title('Data with Multi-Scale Regime Detection')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: States
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Channel States')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[2].plot(scores, 'g-', linewidth=2, alpha=0.8)
    axes[2].set_title('Feature Scores')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Decisions
    decision_map = {'STRONG_BUY': 3, 'BUY': 2, 'HOLD': 1, 'SELL': 0}
    decision_ints = [decision_map.get(d['action'], 0) for d in decisions]
    axes[3].plot(decision_ints, 'purple', linewidth=2, alpha=0.8)
    axes[3].set_title('Decisions')
    axes[3].set_ylabel('Decision')
    axes[3].set_ylim(-0.5, 3.5)
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(['SELL', 'HOLD', 'BUY', 'STRONG_BUY'])
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Regime timeline
    regime_colors = {
        'stable': 'green',
        'transitioning': 'yellow',
        'volatile': 'red',
        'trending': 'blue',
        'mean_reverting': 'purple',
        'unknown': 'gray'
    }
    
    for i, change in enumerate(regime_changes):
        start = change.timestamp
        end = regime_changes[i+1].timestamp if i+1 < len(regime_changes) else len(data)
        
        axes[4].axvspan(start, end, 
                        color=regime_colors.get(change.to_regime.value, 'gray'),
                        alpha=0.3,
                        label=change.to_regime.value if i == 0 else "")
    
    axes[4].set_title('Regime Evolution')
    axes[4].set_xlabel('Sample')
    axes[4].set_ylabel('Regime')
    axes[4].set_ylim(-0.5, 0.5)
    axes[4].set_yticks([])
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_adaptive_multiscale.png', dpi=150, bbox_inches='tight')
    print("Saved: complete_adaptive_multiscale.png")
    
    return states_array, decisions, regime_changes, multiscale


def demonstrate_real_world_application():
    """Demonstrate real-world application: intelligent trading system"""
    print("\n=== Real-World Application: Intelligent Trading System ===")
    
    # Generate realistic financial data
    np.random.seed(42)
    n_days = 1500
    
    # Generate returns with different market regimes
    returns = []
    
    # Bull market (days 0-300)
    returns.extend(np.random.normal(0.002, 0.015, 300))
    
    # Sideways market (days 300-600)
    returns.extend(np.random.normal(0.000, 0.010, 300))
    
    # Bear market (days 600-900)
    returns.extend(np.random.normal(-0.001, 0.020, 300))
    
    # High volatility (days 900-1200)
    returns.extend(np.random.normal(0.000, 0.030, 300))
    
    # Recovery (days 1200-1500)
    returns.extend(np.random.normal(0.001, 0.012, 300))
    
    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create intelligent trading system
    scorer = create_trading_scorer()
    multiscale = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=50,    # ~1 month
        medium_window=200, # ~4 months
        slow_window=1000   # ~2 years
    )
    
    # Create trading decision interpreter
    def trading_interpret(state, regime, score):
        """Enhanced trading decision with regime and score awareness"""
        base_decision = {
            PSI: {'action': 'STRONG_BUY', 'confidence': 0.95},
            PHI: {'action': 'BUY', 'confidence': 0.75},
            DELTA: {'action': 'HOLD', 'confidence': 0.60},
            EMPTY: {'action': 'SELL', 'confidence': 0.80}
        }
        
        decision = base_decision.get(state, {'action': 'HOLD', 'confidence': 0.50})
        
        # Adjust based on regime
        if regime == RegimeType.VOLATILE:
            decision['confidence'] *= 0.8  # Reduce confidence in volatile markets
        elif regime == RegimeType.STABLE:
            decision['confidence'] *= 1.1  # Increase confidence in stable markets
        
        # Adjust based on feature score
        decision['confidence'] *= score
        
        # Add regime-specific reasoning
        decision['reasoning'] = f"State: {state}, Regime: {regime.value}, Score: {score:.3f}"
        
        return decision
    
    # Process financial data
    states = []
    decisions = []
    regime_changes = []
    scores = []
    positions = []
    
    print("Processing financial data with intelligent trading system...")
    
    for i, (price, return_val) in enumerate(zip(prices, returns)):
        # Update multi-scale tracker
        multiscale.update(return_val)
        
        # Check for regime changes
        if multiscale.regime_changed():
            change = multiscale.get_last_regime_change()
            regime_changes.append(change)
            print(f"  Market regime change at day {i}: {change.from_regime.value} → {change.to_regime.value}")
        
        # Encode with regime-appropriate threshold
        state = multiscale.encode_adaptive(return_val)
        states.append(state)
        
        # Score the feature
        if i > 50:
            context = {
                'historical_values': returns[max(0, i-50):i].tolist(),
                'historical_outcomes': returns[max(0, i-50):i].tolist(),
                'sample_size': min(50, i),
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.02
            }
            
            score, dim_scores = scorer.score_and_aggregate(return_val, context)
            scores.append(score)
        else:
            scores.append(0.5)
        
        # Make trading decision
        regime = multiscale.get_current_regime()
        decision = trading_interpret(state, regime, scores[-1])
        decisions.append(decision)
        
        # Convert to position
        if decision['action'] == 'STRONG_BUY':
            position = 1.0
        elif decision['action'] == 'BUY':
            position = 0.5
        elif decision['action'] == 'SELL':
            position = -0.5
        elif decision['action'] == 'STRONG_SELL':
            position = -1.0
        else:
            position = 0.0
        
        positions.append(position)
        
        # Print progress
        if (i + 1) % 300 == 0:
            print(f"  Day {i+1}: regime={regime.value}, state={state}, "
                  f"action={decision['action']}, confidence={decision['confidence']:.3f}")
    
    # Analyze trading performance
    states_array = StateArray(states)
    positions_array = np.array(positions)
    
    # Calculate strategy returns
    strategy_returns = positions_array * returns
    cumulative_returns = np.cumsum(strategy_returns)
    
    # Performance metrics
    total_return = np.sum(strategy_returns)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
    max_drawdown = np.min(np.cumsum(strategy_returns) - np.maximum.accumulate(np.cumsum(strategy_returns)))
    
    print(f"\nTrading System Performance:")
    print(f"  Total days: {len(prices)}")
    print(f"  Market regime changes: {len(regime_changes)}")
    print(f"  Final regime: {multiscale.get_current_regime().value}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Average score: {np.mean(scores):.3f}")
    print(f"  Strategy total return: {total_return:.3f}")
    print(f"  Strategy Sharpe ratio: {sharpe_ratio:.3f}")
    print(f"  Maximum drawdown: {max_drawdown:.3f}")
    
    # Plot trading system results
    fig, axes = plt.subplots(6, 1, figsize=(15, 18))
    
    # Plot 1: Price series
    axes[0].plot(prices, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Price Series with Market Regimes')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Mark regime changes
    for change in regime_changes:
        axes[0].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: Returns
    axes[1].plot(returns, 'g-', linewidth=1, alpha=0.8)
    axes[1].set_title('Daily Returns')
    axes[1].set_ylabel('Return')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: States
    state_ints = states_array.to_ints()
    axes[2].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[2].set_title('Channel States')
    axes[2].set_ylabel('State')
    axes[2].set_ylim(-0.5, 3.5)
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Scores
    axes[3].plot(scores, 'orange', linewidth=2, alpha=0.8)
    axes[3].set_title('Feature Scores')
    axes[3].set_ylabel('Score')
    axes[3].set_ylim(0, 1)
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Positions
    axes[4].plot(positions_array, 'purple', linewidth=2, alpha=0.8)
    axes[4].set_title('Trading Positions')
    axes[4].set_ylabel('Position')
    axes[4].set_ylim(-1.5, 1.5)
    axes[4].grid(True, alpha=0.3)
    
    # Plot 6: Cumulative returns
    axes[5].plot(cumulative_returns, 'red', linewidth=2, alpha=0.8)
    axes[5].set_title('Cumulative Strategy Returns')
    axes[5].set_xlabel('Day')
    axes[5].set_ylabel('Cumulative Return')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_adaptive_trading_system.png', dpi=150, bbox_inches='tight')
    print("Saved: complete_adaptive_trading_system.png")
    
    return states_array, decisions, regime_changes, cumulative_returns


def demonstrate_medical_application():
    """Demonstrate medical application: patient monitoring system"""
    print("\n=== Medical Application: Patient Monitoring System ===")
    
    # Generate realistic medical data
    np.random.seed(42)
    n_hours = 1000
    
    # Generate vital signs with different patient states
    heart_rates = []
    temperatures = []
    blood_pressures = []
    
    # Normal state (hours 0-200)
    heart_rates.extend(np.random.normal(70, 5, 200))
    temperatures.extend(np.random.normal(37.0, 0.2, 200))
    blood_pressures.extend(np.random.normal(120, 10, 200))
    
    # Fever state (hours 200-400)
    heart_rates.extend(np.random.normal(85, 8, 200))
    temperatures.extend(np.random.normal(38.5, 0.3, 200))
    blood_pressures.extend(np.random.normal(130, 15, 200))
    
    # Critical state (hours 400-600)
    heart_rates.extend(np.random.normal(110, 15, 200))
    temperatures.extend(np.random.normal(39.5, 0.5, 200))
    blood_pressures.extend(np.random.normal(150, 20, 200))
    
    # Recovery state (hours 600-800)
    heart_rates.extend(np.random.normal(75, 6, 200))
    temperatures.extend(np.random.normal(37.2, 0.3, 200))
    blood_pressures.extend(np.random.normal(125, 12, 200))
    
    # Stable state (hours 800-1000)
    heart_rates.extend(np.random.normal(68, 4, 200))
    temperatures.extend(np.random.normal(36.8, 0.2, 200))
    blood_pressures.extend(np.random.normal(118, 8, 200))
    
    # Create medical monitoring system
    scorer = create_medical_scorer()
    multiscale = MultiScaleAdaptiveThreshold(
        use_topology=True,
        fast_window=50,   # ~2 hours
        medium_window=200, # ~8 hours
        slow_window=500    # ~20 hours
    )
    
    # Create medical decision interpreter
    def medical_interpret(state, regime, score):
        """Medical decision interpreter"""
        if state == PSI:
            return {
                'action': 'CRITICAL_ALERT',
                'confidence': 0.95,
                'reasoning': 'Critical vital signs detected',
                'urgency': 'HIGH'
            }
        elif state == PHI:
            return {
                'action': 'MONITOR_CLOSELY',
                'confidence': 0.75,
                'reasoning': 'Elevated vital signs',
                'urgency': 'MEDIUM'
            }
        elif state == DELTA:
            return {
                'action': 'ROUTINE_CHECK',
                'confidence': 0.60,
                'reasoning': 'Normal vital signs',
                'urgency': 'LOW'
            }
        else:  # EMPTY
            return {
                'action': 'NORMAL',
                'confidence': 0.80,
                'reasoning': 'Stable vital signs',
                'urgency': 'LOW'
            }
    
    # Process medical data
    states = []
    decisions = []
    regime_changes = []
    scores = []
    
    print("Processing medical data with patient monitoring system...")
    
    for i in range(n_hours):
        # Combine vital signs into composite score
        hr_score = (heart_rates[i] - 70) / 20  # Normalize
        temp_score = (temperatures[i] - 37.0) / 2.0  # Normalize
        bp_score = (blood_pressures[i] - 120) / 30  # Normalize
        
        composite_score = (hr_score + temp_score + bp_score) / 3
        
        # Update multi-scale tracker
        multiscale.update(composite_score)
        
        # Check for regime changes
        if multiscale.regime_changed():
            change = multiscale.get_last_regime_change()
            regime_changes.append(change)
            print(f"  Patient state change at hour {i}: {change.from_regime.value} → {change.to_regime.value}")
        
        # Encode with regime-appropriate threshold
        state = multiscale.encode_adaptive(composite_score)
        states.append(state)
        
        # Score the feature
        if i > 50:
            context = {
                'historical_values': [composite_score] * 50,  # Simplified
                'historical_outcomes': [composite_score] * 50,
                'sample_size': 50,
                'age_seconds': 0,
                'missing_rate': 0.01,
                'noise_level': 0.1
            }
            
            score, dim_scores = scorer.score_and_aggregate(composite_score, context)
            scores.append(score)
        else:
            scores.append(0.5)
        
        # Make medical decision
        regime = multiscale.get_current_regime()
        decision = medical_interpret(state, regime, scores[-1])
        decisions.append(decision)
        
        # Print progress
        if (i + 1) % 200 == 0:
            print(f"  Hour {i+1}: regime={regime.value}, state={state}, "
                  f"action={decision['action']}, urgency={decision['urgency']}")
    
    # Analyze medical results
    states_array = StateArray(states)
    
    print(f"\nMedical Monitoring Results:")
    print(f"  Total hours: {n_hours}")
    print(f"  Patient state changes: {len(regime_changes)}")
    print(f"  Final state: {multiscale.get_current_regime().value}")
    print(f"  State distribution: {states_array.count_by_state()}")
    print(f"  Average score: {np.mean(scores):.3f}")
    
    # Medical decision analysis
    actions = [d['action'] for d in decisions]
    action_counts = {}
    for action in actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"  Medical decision distribution: {action_counts}")
    
    # Plot medical monitoring results
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    
    # Plot 1: Vital signs
    axes[0].plot(heart_rates, 'r-', label='Heart Rate', alpha=0.7)
    axes[0].plot(temperatures * 10, 'b-', label='Temperature (×10)', alpha=0.7)
    axes[0].plot(blood_pressures, 'g-', label='Blood Pressure', alpha=0.7)
    axes[0].set_title('Vital Signs Over Time')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mark regime changes
    for change in regime_changes:
        axes[0].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
    
    # Plot 2: States
    state_ints = states_array.to_ints()
    axes[1].plot(state_ints, 'o-', markersize=2, alpha=0.7)
    axes[1].set_title('Patient State (Channel States)')
    axes[1].set_ylabel('State')
    axes[1].set_ylim(-0.5, 3.5)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(['∅', 'δ', 'φ', 'ψ'])
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Scores
    axes[2].plot(scores, 'orange', linewidth=2, alpha=0.8)
    axes[2].set_title('Medical Feature Scores')
    axes[2].set_ylabel('Score')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Medical decisions
    decision_map = {'CRITICAL_ALERT': 3, 'MONITOR_CLOSELY': 2, 'ROUTINE_CHECK': 1, 'NORMAL': 0}
    decision_ints = [decision_map.get(d['action'], 0) for d in decisions]
    axes[3].plot(decision_ints, 'purple', linewidth=2, alpha=0.8)
    axes[3].set_title('Medical Decisions')
    axes[3].set_ylabel('Decision')
    axes[3].set_ylim(-0.5, 3.5)
    axes[3].set_yticks([0, 1, 2, 3])
    axes[3].set_yticklabels(['NORMAL', 'ROUTINE_CHECK', 'MONITOR_CLOSELY', 'CRITICAL_ALERT'])
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Patient state timeline
    regime_colors = {
        'stable': 'green',
        'transitioning': 'yellow',
        'volatile': 'red',
        'trending': 'blue',
        'mean_reverting': 'purple',
        'unknown': 'gray'
    }
    
    for i, change in enumerate(regime_changes):
        start = change.timestamp
        end = regime_changes[i+1].timestamp if i+1 < len(regime_changes) else n_hours
        
        axes[4].axvspan(start, end, 
                        color=regime_colors.get(change.to_regime.value, 'gray'),
                        alpha=0.3,
                        label=change.to_regime.value if i == 0 else "")
    
    axes[4].set_title('Patient State Evolution')
    axes[4].set_xlabel('Hour')
    axes[4].set_ylabel('Patient State')
    axes[4].set_ylim(-0.5, 0.5)
    axes[4].set_yticks([])
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('complete_adaptive_medical_system.png', dpi=150, bbox_inches='tight')
    print("Saved: complete_adaptive_medical_system.png")
    
    return states_array, decisions, regime_changes


def main():
    """Main complete adaptive system example"""
    print("ChannelPy Complete Adaptive System Example")
    print("=" * 60)
    
    # 1. Basic integration
    basic_states, basic_decisions, basic_scores, basic_scorer = demonstrate_basic_integration()
    
    # 2. Multi-scale integration
    multiscale_states, multiscale_decisions, multiscale_regimes, multiscale_tracker = demonstrate_multiscale_integration()
    
    # 3. Real-world trading application
    trading_states, trading_decisions, trading_regimes, trading_returns = demonstrate_real_world_application()
    
    # 4. Medical application
    medical_states, medical_decisions, medical_regimes = demonstrate_medical_application()
    
    print("\n" + "=" * 60)
    print("Complete adaptive system example completed successfully!")
    print("\nGenerated files:")
    print("- complete_adaptive_basic.png: Basic adaptive system integration")
    print("- complete_adaptive_multiscale.png: Multi-scale adaptive system")
    print("- complete_adaptive_trading_system.png: Intelligent trading system")
    print("- complete_adaptive_medical_system.png: Patient monitoring system")
    print("\nKey features demonstrated:")
    print("- Feature scoring for multi-dimensional evaluation")
    print("- Topology-aware adaptive thresholds")
    print("- Multi-scale regime detection")
    print("- Intelligent decision making")
    print("- Real-world applications in trading and medical monitoring")
    print("\nNext steps:")
    print("- Use the complete adaptive system for your data")
    print("- Customize scorers for your domain")
    print("- Implement intelligent decision making")
    print("- Apply to real-world streaming data")
    print("\nExample usage:")
    print("""
from channelpy.adaptive import (
    FeatureScorer, 
    TopologyAdaptiveThreshold,
    MultiScaleAdaptiveThreshold,
    create_trading_scorer
)

# Create scorer
scorer = create_trading_scorer()

# Create topology-aware tracker
topology_threshold = TopologyAdaptiveThreshold(
    window_size=1000,
    feature_scorer=scorer
)

# Or use multi-scale for regime detection
multiscale = MultiScaleAdaptiveThreshold(use_topology=True)

# Process stream
for value in data_stream:
    multiscale.update(value)
    
    # Check regime
    if multiscale.regime_changed():
        print(f"Regime change: {multiscale.get_current_regime()}")
    
    # Encode with regime-appropriate threshold
    state = multiscale.encode_adaptive(value)
    
    # Make decision based on state
    decision = interpret(state)
""")


if __name__ == "__main__":
    main()




