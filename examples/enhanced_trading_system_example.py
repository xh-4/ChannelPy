"""
Enhanced Trading System Example

This example demonstrates the updated trading system with advanced adaptive components:
- Topology-aware adaptive thresholds
- Multi-scale regime detection
- Intelligent feature scoring
- Regime-aware decision making
- Risk management with regime awareness

This showcases the complete integration of all adaptive components in a real-world trading application.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

# Import the enhanced trading system
from channelpy.applications.trading import (
    TradingChannelSystem, LiveTradingSystem, TechnicalIndicators
)
from channelpy.adaptive import create_trading_scorer
from channelpy import StateArray, EMPTY, DELTA, PHI, PSI


def generate_realistic_trading_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate realistic trading data with multiple market regimes"""
    np.random.seed(42)
    
    # Generate returns with different market regimes
    returns = []
    
    # Bull market (days 0-200)
    returns.extend(np.random.normal(0.001, 0.015, 200))
    
    # Sideways market (days 200-400)
    returns.extend(np.random.normal(0.000, 0.010, 200))
    
    # Bear market (days 400-600)
    returns.extend(np.random.normal(-0.001, 0.020, 200))
    
    # High volatility (days 600-800)
    returns.extend(np.random.normal(0.000, 0.030, 200))
    
    # Recovery (days 800-1000)
    returns.extend(np.random.normal(0.001, 0.012, 200))
    
    returns = np.array(returns)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (price, return_val) in enumerate(zip(prices, returns)):
        # Generate realistic OHLC from price and return
        high = price * (1 + abs(return_val) * 0.5)
        low = price * (1 - abs(return_val) * 0.5)
        open_price = price * (1 - return_val * 0.3)
        close = price
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000000
        volatility_multiplier = 1 + abs(return_val) * 10
        volume = int(base_volume * volatility_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    
    return df


def demonstrate_basic_vs_advanced_systems():
    """Compare basic vs advanced adaptive trading systems"""
    print("=== Basic vs Advanced Trading Systems Comparison ===")
    
    # Generate data
    df = generate_realistic_trading_data(500)
    
    # Create basic system
    basic_system = TradingChannelSystem(
        strategy='simple',
        use_advanced_adaptive=False
    )
    
    # Create advanced system
    advanced_system = TradingChannelSystem(
        strategy='simple',
        use_advanced_adaptive=True
    )
    
    # Fit both systems
    print("Fitting systems...")
    basic_system.fit(df[:300])  # Use first 300 days for training
    advanced_system.fit(df[:300])
    
    # Test on remaining data
    test_df = df[300:]
    
    # Process data with both systems
    basic_signals = []
    advanced_signals = []
    regime_changes = []
    
    print("Processing test data...")
    for i, (idx, row) in enumerate(test_df.iterrows()):
        # Basic system
        basic_signal = basic_system.process_bar(row)
        basic_signals.append(basic_signal)
        
        # Advanced system
        advanced_signal = advanced_system.process_bar(row)
        advanced_signals.append(advanced_signal)
        
        # Check for regime changes
        if advanced_system.use_advanced_adaptive:
            regime_info = advanced_system.encoder.get_regime_info()
            if regime_info['regime_changes'] > len(regime_changes):
                regime_changes.append({
                    'timestamp': idx,
                    'regime': regime_info['current_regime'],
                    'confidence': regime_info['regime_confidence']
                })
        
        # Print progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} days")
    
    # Analyze results
    print(f"\nBasic System Results:")
    print(f"  Total signals: {len(basic_signals)}")
    basic_actions = [s['action'] for s in basic_signals]
    basic_action_counts = {action: basic_actions.count(action) for action in set(basic_actions)}
    print(f"  Signal distribution: {basic_action_counts}")
    
    print(f"\nAdvanced System Results:")
    print(f"  Total signals: {len(advanced_signals)}")
    advanced_actions = [s['action'] for s in advanced_signals]
    advanced_action_counts = {action: advanced_actions.count(action) for action in set(advanced_actions)}
    print(f"  Signal distribution: {advanced_action_counts}")
    
    print(f"\nRegime Changes Detected: {len(regime_changes)}")
    for change in regime_changes:
        print(f"  {change['timestamp'].strftime('%Y-%m-%d')}: {change['regime']} (confidence: {change['confidence']:.3f})")
    
    # Get system information
    basic_info = basic_system.get_system_info()
    advanced_info = advanced_system.get_system_info()
    
    print(f"\nBasic System Info:")
    print(f"  Use advanced adaptive: {basic_info['use_advanced_adaptive']}")
    print(f"  Strategy: {basic_info['strategy_name']}")
    
    print(f"\nAdvanced System Info:")
    print(f"  Use advanced adaptive: {advanced_info['use_advanced_adaptive']}")
    print(f"  Strategy: {advanced_info['strategy_name']}")
    print(f"  Current regime: {advanced_info['regime_info']['current_regime']}")
    print(f"  Regime confidence: {advanced_info['regime_info']['regime_confidence']:.3f}")
    print(f"  Total regime changes: {advanced_info['regime_info']['regime_changes']}")
    
    return basic_system, advanced_system, basic_signals, advanced_signals, regime_changes


def demonstrate_live_trading_system():
    """Demonstrate live trading system with regime awareness"""
    print("\n=== Live Trading System with Regime Awareness ===")
    
    # Generate data
    df = generate_realistic_trading_data(300)
    
    # Create live trading system
    live_system = LiveTradingSystem(
        strategy='adaptive',
        risk_per_trade=0.02,
        use_advanced_adaptive=True
    )
    
    # Fit system
    print("Fitting live trading system...")
    live_system.fit(df[:200])  # Use first 200 days for training
    
    # Simulate live trading
    print("Simulating live trading...")
    current_capital = 100000.0  # $100,000 starting capital
    positions = []
    regime_history = []
    
    for i, (idx, row) in enumerate(df[200:]):
        # Process bar in live trading
        signal = live_system.process_bar_live(row.to_dict(), current_capital)
        
        # Record position
        positions.append({
            'timestamp': idx,
            'price': row['close'],
            'signal': signal,
            'capital': current_capital
        })
        
        # Track regime changes
        if 'regime_info' in signal:
            regime_history.append({
                'timestamp': idx,
                'regime': signal['regime_info']['current_regime'],
                'confidence': signal['regime_info']['regime_confidence']
            })
        
        # Simulate trade execution
        if signal['action'] == 'BUY' and signal.get('size', 0) > 0:
            # Execute buy order
            cost = row['close'] * signal['size']
            if current_capital >= cost:
                current_capital -= cost
                print(f"  BUY: {signal['size']:.2f} shares at ${row['close']:.2f} (cost: ${cost:.2f})")
                if 'regime_info' in signal:
                    print(f"    Regime: {signal['regime_info']['current_regime']} (confidence: {signal['regime_info']['regime_confidence']:.3f})")
        
        elif signal['action'] == 'SELL':
            # Execute sell order (simplified)
            print(f"  SELL signal at ${row['close']:.2f}")
            if 'regime_info' in signal:
                print(f"    Regime: {signal['regime_info']['current_regime']} (confidence: {signal['regime_info']['regime_confidence']:.3f})")
        
        # Print progress
        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1} days, capital: ${current_capital:.2f}")
    
    # Analyze results
    print(f"\nLive Trading Results:")
    print(f"  Starting capital: $100,000")
    print(f"  Final capital: ${current_capital:.2f}")
    print(f"  Total return: {((current_capital - 100000) / 100000) * 100:.2f}%")
    print(f"  Regime changes detected: {len(regime_history)}")
    
    # Show regime evolution
    if regime_history:
        print(f"\nRegime Evolution:")
        for regime in regime_history[-5:]:  # Show last 5 regime changes
            print(f"  {regime['timestamp'].strftime('%Y-%m-%d')}: {regime['regime']} (confidence: {regime['confidence']:.3f})")
    
    return live_system, positions, regime_history


def demonstrate_regime_aware_backtesting():
    """Demonstrate regime-aware backtesting"""
    print("\n=== Regime-Aware Backtesting ===")
    
    # Generate data
    df = generate_realistic_trading_data(800)
    
    # Create systems for comparison
    basic_system = TradingChannelSystem(strategy='simple', use_advanced_adaptive=False)
    advanced_system = TradingChannelSystem(strategy='simple', use_advanced_adaptive=True)
    
    # Fit systems
    print("Fitting systems for backtesting...")
    basic_system.fit(df[:400])  # Use first 400 days for training
    advanced_system.fit(df[:400])
    
    # Backtest on remaining data
    test_df = df[400:]
    
    print("Running backtests...")
    basic_results = basic_system.backtest(test_df, initial_capital=100000.0)
    advanced_results = advanced_system.backtest(test_df, initial_capital=100000.0)
    
    # Compare results
    print(f"\nBacktest Results Comparison:")
    print(f"  Basic System:")
    print(f"    Total return: {basic_results['total_return']:.2%}")
    print(f"    Sharpe ratio: {basic_results['sharpe_ratio']:.3f}")
    print(f"    Max drawdown: {basic_results['max_drawdown']:.2%}")
    print(f"    Number of trades: {basic_results['num_trades']}")
    
    print(f"  Advanced System:")
    print(f"    Total return: {advanced_results['total_return']:.2%}")
    print(f"    Sharpe ratio: {advanced_results['sharpe_ratio']:.3f}")
    print(f"    Max drawdown: {advanced_results['max_drawdown']:.2%}")
    print(f"    Number of trades: {advanced_results['num_trades']}")
    
    # Get advanced system info
    advanced_info = advanced_system.get_system_info()
    print(f"\nAdvanced System Regime Info:")
    print(f"  Current regime: {advanced_info['regime_info']['current_regime']}")
    print(f"  Regime confidence: {advanced_info['regime_info']['regime_confidence']:.3f}")
    print(f"  Total regime changes: {advanced_info['regime_info']['regime_changes']}")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Price data
    axes[0].plot(test_df.index, test_df['close'], 'b-', linewidth=1, alpha=0.8)
    axes[0].set_title('Price Data for Backtesting')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Equity curves
    basic_equity = basic_results['equity_curve']
    advanced_equity = advanced_results['equity_curve']
    
    axes[1].plot(test_df.index, basic_equity, 'r-', label='Basic System', linewidth=2)
    axes[1].plot(test_df.index, advanced_equity, 'g-', label='Advanced System', linewidth=2)
    axes[1].set_title('Equity Curves Comparison')
    axes[1].set_ylabel('Equity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Regime information
    if advanced_info['regime_info']['regime_changes'] > 0:
        regime_changes = advanced_system.encoder.regime_changes
        for change in regime_changes:
            axes[2].axvline(change.timestamp, color='red', linestyle='--', alpha=0.7)
        
        axes[2].set_title('Regime Changes Detected')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Regime Changes')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_trading_system_backtest.png', dpi=150, bbox_inches='tight')
    print("Saved: enhanced_trading_system_backtest.png")
    
    return basic_results, advanced_results, advanced_info


def demonstrate_topology_analysis():
    """Demonstrate topology analysis in trading"""
    print("\n=== Topology Analysis in Trading ===")
    
    # Generate data
    df = generate_realistic_trading_data(400)
    
    # Create advanced system
    system = TradingChannelSystem(strategy='adaptive', use_advanced_adaptive=True)
    
    # Fit system
    print("Fitting system for topology analysis...")
    system.fit(df)
    
    # Get topology information
    topology_info = system.encoder.get_topology_info()
    
    print(f"\nTopology Analysis Results:")
    for channel, topology in topology_info.items():
        print(f"  {channel.upper()} Channel:")
        print(f"    Modality: {topology.modality}")
        print(f"    Skewness: {topology.skewness:.3f}")
        print(f"    Kurtosis: {topology.kurtosis:.3f}")
        print(f"    Gaps detected: {topology.gaps}")
        print(f"    Density variance: {topology.density_variance:.3f}")
        print(f"    Components: {topology.components}")
        print(f"    Strategy: {topology.strategy}")
    
    # Get regime information
    regime_info = system.encoder.get_regime_info()
    print(f"\nRegime Information:")
    print(f"  Current regime: {regime_info['current_regime']}")
    print(f"  Regime confidence: {regime_info['regime_confidence']:.3f}")
    print(f"  Total regime changes: {regime_info['regime_changes']}")
    
    # Get system statistics
    system_info = system.get_system_info()
    print(f"\nSystem Statistics:")
    print(f"  Use advanced adaptive: {system_info['use_advanced_adaptive']}")
    print(f"  Strategy: {system_info['strategy_name']}")
    print(f"  Scorer dimensions: {system_info['scorer_stats']['dimensions']}")
    print(f"  Total scores recorded: {system_info['scorer_stats']['total_scores']}")
    
    return system, topology_info, regime_info


def main():
    """Main enhanced trading system example"""
    print("ChannelPy Enhanced Trading System Example")
    print("=" * 60)
    
    # 1. Compare basic vs advanced systems
    basic_system, advanced_system, basic_signals, advanced_signals, regime_changes = demonstrate_basic_vs_advanced_systems()
    
    # 2. Demonstrate live trading system
    live_system, positions, regime_history = demonstrate_live_trading_system()
    
    # 3. Demonstrate regime-aware backtesting
    basic_results, advanced_results, advanced_info = demonstrate_regime_aware_backtesting()
    
    # 4. Demonstrate topology analysis
    system, topology_info, regime_info = demonstrate_topology_analysis()
    
    print("\n" + "=" * 60)
    print("Enhanced trading system example completed successfully!")
    print("\nKey features demonstrated:")
    print("- Topology-aware adaptive thresholds for intelligent threshold selection")
    print("- Multi-scale regime detection for robust market analysis")
    print("- Intelligent feature scoring for enhanced decision making")
    print("- Regime-aware risk management for optimal position sizing")
    print("- Real-time regime change detection and adaptation")
    print("- Comprehensive system information and monitoring")
    print("\nGenerated files:")
    print("- enhanced_trading_system_backtest.png: Backtesting results comparison")
    print("\nNext steps:")
    print("- Use the enhanced trading system for your data")
    print("- Customize regime detection parameters")
    print("- Implement custom trading strategies")
    print("- Apply to live trading with real-time data")
    print("\nExample usage:")
    print("""
# Create enhanced trading system
system = TradingChannelSystem(
    strategy='adaptive',
    use_advanced_adaptive=True
)

# Fit on historical data
system.fit(historical_data)

# Process real-time data
for bar in live_data_stream:
    signal = system.process_bar(bar)
    
    # Check for regime changes
    regime_info = system.encoder.get_regime_info()
    if regime_info['regime_changes'] > 0:
        print(f"Regime change detected: {regime_info['current_regime']}")
    
    # Make trading decision
    if signal['action'] != 'HOLD':
        execute_trade(signal)
""")


if __name__ == "__main__":
    main()




