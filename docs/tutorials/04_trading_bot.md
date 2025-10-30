# Tutorial 04: Building a Complete Trading Bot

In this tutorial, we'll build a complete trading system using ChannelPy's topology-aware adaptive thresholds. This demonstrates how channel algebra can create interpretable, robust trading signals.

## What We'll Build

A trading bot that:
- Processes real-time price and volume data
- Adapts thresholds to market regime changes
- Detects topology shifts (trending → ranging → volatile)
- Generates buy/sell/hold signals with explanations
- Tracks performance and explains decisions

## Prerequisites
```python
import numpy as np
import pandas as pd
from channelpy import State, PSI, DELTA, PHI, EMPTY
from channelpy.applications import TradingChannelSystem
from channelpy.adaptive import MultiScaleAdaptiveThreshold, create_trading_scorer
from channelpy.visualization import plot_states, plot_threshold_adaptation
```

## Step 1: Load Market Data
```python
# For this tutorial, we'll use synthetic data
# In production, replace with real data from your broker API

def generate_market_data(n_samples=1000, regime='normal'):
    """
    Generate synthetic market data
    
    regime: 'normal', 'trending', 'volatile', 'ranging'
    """
    np.random.seed(42)
    
    if regime == 'normal':
        # Normal market: random walk with drift
        returns = np.random.randn(n_samples) * 0.01 + 0.0005
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 0.5, n_samples)
    
    elif regime == 'trending':
        # Strong trend
        trend = np.linspace(0, 0.3, n_samples)
        noise = np.random.randn(n_samples) * 0.005
        returns = trend / n_samples + noise
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 0.3, n_samples)
    
    elif regime == 'volatile':
        # High volatility
        returns = np.random.randn(n_samples) * 0.03
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 0.8, n_samples)
    
    elif regime == 'ranging':
        # Mean-reverting
        prices = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples))
        prices += np.random.randn(n_samples) * 0.5
        volumes = np.random.lognormal(10, 0.4, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'price': prices,
        'volume': volumes
    })
    
    return df

# Generate data with regime changes
data_normal = generate_market_data(300, 'normal')
data_trending = generate_market_data(300, 'trending')
data_volatile = generate_market_data(200, 'volatile')
data_ranging = generate_market_data(200, 'ranging')

market_data = pd.concat([
    data_normal, 
    data_trending, 
    data_volatile, 
    data_ranging
], ignore_index=True)

print(f"Generated {len(market_data)} market ticks")
```

## Step 2: Initialize Trading System
```python
from channelpy.applications.trading import TradingChannelSystem

# Create trading system with topology-aware thresholds
trading_system = TradingChannelSystem(
    use_topology=True,
    use_multiscale=True,
    fast_window=50,
    medium_window=200,
    slow_window=500
)

# Warm up with historical data
warmup_data = market_data.iloc[:100]
trading_system.fit(
    prices=warmup_data['price'],
    volumes=warmup_data['volume']
)

print("Trading system initialized")
print(f"Initial regime: {trading_system.get_regime()}")
```

## Step 3: Process Market Stream
```python
# Storage for results
results = []

# Process each tick
for idx in range(100, len(market_data)):
    price = market_data.iloc[idx]['price']
    volume = market_data.iloc[idx]['volume']
    
    # Process tick
    signal = trading_system.process_tick(price, volume)
    
    # Store results
    results.append({
        'index': idx,
        'price': price,
        'volume': volume,
        'action': signal['action'],
        'confidence': signal['confidence'],
        'price_state': signal['states']['price'],
        'volume_state': signal['states']['volume'],
        'regime': signal['regime'],
        'explanation': signal['explanation']
    })
    
    # Print regime changes
    if idx > 100 and results[-1]['regime'] != results[-2]['regime']:
        print(f"\n📊 REGIME CHANGE at tick {idx}")
        print(f"   {results[-2]['regime']} → {results[-1]['regime']}")
        print(f"   Price: {price:.2f}")

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)
print(f"\nProcessed {len(results_df)} ticks")
```

## Step 4: Analyze Signals
```python
# Count actions
action_counts = results_df['action'].value_counts()
print("\n📈 Signal Distribution:")
print(action_counts)

# Average confidence by action
avg_confidence = results_df.groupby('action')['confidence'].mean()
print("\n🎯 Average Confidence:")
print(avg_confidence)

# State distribution
print("\n🔄 Price State Distribution:")
print(results_df['price_state'].value_counts())

print("\n📊 Volume State Distribution:")
print(results_df['volume_state'].value_counts())
```

## Step 5: Backtest Performance
```python
def backtest(results_df, initial_capital=10000):
    """
    Simple backtest: buy/sell 1 share on signals
    """
    capital = initial_capital
    position = 0  # Number of shares
    trades = []
    
    for idx, row in results_df.iterrows():
        price = row['price']
        action = row['action']
        confidence = row['confidence']
        
        # Only trade on high-confidence signals
        if confidence < 0.7:
            continue
        
        if action == 'BUY' and position == 0:
            # Buy 1 share
            shares_to_buy = capital // price
            if shares_to_buy > 0:
                position = shares_to_buy
                cost = shares_to_buy * price
                capital -= cost
                trades.append({
                    'index': idx,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'capital': capital,
                    'position_value': position * price
                })
        
        elif action == 'SELL' and position > 0:
            # Sell all shares
            proceeds = position * price
            capital += proceeds
            trades.append({
                'index': idx,
                'action': 'SELL',
                'price': price,
                'shares': position,
                'capital': capital,
                'position_value': 0
            })
            position = 0
    
    # Close any open position
    if position > 0:
        final_price = results_df.iloc[-1]['price']
        capital += position * final_price
        position = 0
    
    trades_df = pd.DataFrame(trades)
    
    return {
        'final_capital': capital,
        'return': (capital - initial_capital) / initial_capital * 100,
        'num_trades': len(trades_df),
        'trades': trades_df
    }

# Run backtest
backtest_results = backtest(results_df)

print("\n💰 Backtest Results:")
print(f"Initial Capital: $10,000")
print(f"Final Capital: ${backtest_results['final_capital']:.2f}")
print(f"Return: {backtest_results['return']:.2f}%")
print(f"Number of Trades: {backtest_results['num_trades']}")

# Buy and hold comparison
buy_hold_return = (
    (results_df.iloc[-1]['price'] - results_df.iloc[0]['price']) / 
    results_df.iloc[0]['price'] * 100
)
print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
```

## Step 6: Visualize Results
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: Price with buy/sell signals
ax1 = axes[0]
ax1.plot(results_df['index'], results_df['price'], 
         label='Price', linewidth=1, color='black', alpha=0.7)

# Mark buy signals
buy_signals = results_df[results_df['action'] == 'BUY']
ax1.scatter(buy_signals['index'], buy_signals['price'], 
           marker='^', color='green', s=100, label='BUY', zorder=5)

# Mark sell signals
sell_signals = results_df[results_df['action'] == 'SELL']
ax1.scatter(sell_signals['index'], sell_signals['price'], 
           marker='v', color='red', s=100, label='SELL', zorder=5)

ax1.set_ylabel('Price')
ax1.set_title('Trading Signals')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: State evolution
ax2 = axes[1]
price_state_int = results_df['price_state'].apply(
    lambda s: s.to_int() if hasattr(s, 'to_int') else 0
)
ax2.plot(results_df['index'], price_state_int, 
         label='Price State', drawstyle='steps-post')
ax2.set_ylabel('State')
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(['∅', 'φ', 'δ', 'ψ'])
ax2.set_title('Price State Evolution')
ax2.grid(True, alpha=0.3)

# Plot 3: Confidence over time
ax3 = axes[2]
ax3.plot(results_df['index'], results_df['confidence'], 
         label='Signal Confidence', color='purple')
ax3.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Trade Threshold')
ax3.set_ylabel('Confidence')
ax3.set_title('Signal Confidence')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Regime detection
ax4 = axes[3]
regime_map = {
    'STABLE': 0,
    'TRANSITIONING': 1,
    'VOLATILE': 2,
    'TRENDING': 3,
    'MEAN_REVERTING': 4
}
regime_int = results_df['regime'].map(regime_map).fillna(0)

# Color by regime
colors = ['green', 'yellow', 'red', 'blue', 'purple']
for i in range(len(regime_int) - 1):
    ax4.axvspan(
        results_df.iloc[i]['index'],
        results_df.iloc[i+1]['index'],
        color=colors[int(regime_int.iloc[i]) % len(colors)],
        alpha=0.3
    )

ax4.set_ylabel('Regime')
ax4.set_xlabel('Time (ticks)')
ax4.set_title('Market Regime Detection')
ax4.set_yticks(list(regime_map.values()))
ax4.set_yticklabels(list(regime_map.keys()), fontsize=8)

plt.tight_layout()
plt.savefig('trading_bot_results.png', dpi=150)
plt.show()
```

## Step 7: Understanding Decisions
```python
# Print detailed explanation for a few trades
print("\n📝 Sample Trade Explanations:\n")

for idx in [5, 10, 15]:
    if idx < len(results_df):
        row = results_df.iloc[idx]
        print(f"Tick {row['index']}:")
        print(f"  Price: ${row['price']:.2f}")
        print(f"  Action: {row['action']}")
        print(f"  Confidence: {row['confidence']:.2%}")
        print(f"  Price State: {row['price_state']}")
        print(f"  Volume State: {row['volume_state']}")
        print(f"  Regime: {row['regime']}")
        print(f"  Explanation: {row['explanation']}")
        print()
```

## Key Insights

### Why This Works

1. **Topology Awareness**
   - Detects when market structure changes (trending → ranging)
   - Adapts thresholds to current regime
   - Avoids false signals during regime transitions

2. **Multi-Scale Analysis**
   - Fast scale catches quick moves
   - Slow scale provides stable baseline
   - Divergence signals regime changes

3. **Interpretability**
   - Every decision has an explanation
   - States show exactly why signal triggered
   - Confidence quantifies signal quality

4. **Robustness**
   - No manual parameter tuning
   - Adapts to volatility automatically
   - Graceful degradation in poor conditions

### Common Patterns

**ψ + ψ → Strong Buy**
```
Price State: ψ (present and member = above both thresholds)
Volume State: ψ (strong volume confirmation)
→ High confidence buy signal
```

**δ + φ → Weak Signal**
```
Price State: δ (above i threshold, but not q = weak signal)
Volume State: φ (expected but not present = no volume)
→ Low confidence, usually filtered out
```

**∅ + ψ → Potential Sell**
```
Price State: ∅ (below both thresholds = weak)
Volume State: ψ (but volume is high = distribution?)
→ Possible panic selling, investigate
```

## Production Considerations

### 1. Data Quality
```python
# Add data validation
def validate_tick(price, volume):
    if price <= 0:
        raise ValueError("Invalid price")
    if volume < 0:
        raise ValueError("Invalid volume")
    if not np.isfinite(price) or not np.isfinite(volume):
        raise ValueError("Non-finite values")
```

### 2. Risk Management
```python
# Add position sizing
def calculate_position_size(signal, capital, risk_per_trade=0.02):
    """
    Size position based on confidence and risk tolerance
    """
    confidence = signal['confidence']
    max_position = capital * risk_per_trade
    position_size = max_position * confidence
    return position_size
```

### 3. Transaction Costs
```python
# Include fees and slippage
def apply_costs(trade_value, commission_rate=0.001, slippage_bps=5):
    commission = trade_value * commission_rate
    slippage = trade_value * (slippage_bps / 10000)
    return commission + slippage
```

### 4. Logging and Monitoring
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trading_bot')

# Log important events
logger.info(f"Signal generated: {signal['action']} at {price}")
logger.warning(f"Regime change detected: {old_regime} → {new_regime}")
logger.error(f"Trade execution failed: {error}")
```

## Next Steps

- **Tutorial 05**: Apply similar techniques to medical diagnosis
- **How-To Guide**: Create custom encoders for your market indicators
- **API Reference**: Deep dive into TradingChannelSystem class

## Complete Code

The complete, runnable code is available in `examples/trading_bot.py`.

---

**💡 Key Takeaway**: Channel algebra makes trading signals interpretable while remaining adaptive to market changes. Every decision can be explained in terms of channel states and topology.