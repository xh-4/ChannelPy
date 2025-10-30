# Applications API Reference

The applications module provides domain-specific implementations of channel algebra for trading, medical diagnosis, signal processing, and more. Each application demonstrates best practices and can serve as a template for new domains.

## Overview
```python
from channelpy.applications import (
    # Trading
    TradingChannelSystem,
    TechnicalIndicatorEncoder,
    TradingSignalInterpreter,
    
    # Medical
    MedicalDiagnosisSystem,
    SymptomEncoder,
    TestResultEncoder,
    DiagnosisInterpreter,
    
    # Signal Processing
    SignalChannelSystem,
    FrequencyEncoder,
    AnomalyDetector,
    
    # Utilities
    create_application,
    ApplicationTemplate
)
```

---

## Trading Applications

### `TradingChannelSystem`

Complete trading system using channel algebra.
```python
class TradingChannelSystem:
    """
    End-to-end trading system with channel algebra
    
    Features:
    - Multi-indicator encoding (price, volume, volatility)
    - Topology-aware adaptive thresholds
    - Regime detection
    - Risk-aware position sizing
    - Explainable signals
    
    Examples
    --------
    >>> import pandas as pd
    >>> from channelpy.applications import TradingChannelSystem
    >>> 
    >>> # Load historical data
    >>> df = pd.read_csv('AAPL.csv')
    >>> 
    >>> # Create system
    >>> system = TradingChannelSystem()
    >>> system.fit(df['Close'], df['Volume'])
    >>> 
    >>> # Process new data
    >>> signal = system.process_tick(
    ...     price=150.0,
    ...     volume=1000000,
    ...     timestamp=pd.Timestamp.now()
    ... )
    >>> 
    >>> print(signal['action'])      # 'BUY', 'SELL', or 'HOLD'
    >>> print(signal['confidence'])  # 0.0 to 1.0
    >>> print(signal['reason'])      # Human-readable explanation
    """
```

#### Constructor
```python
def __init__(
    self,
    use_topology: bool = True,
    use_multiscale: bool = True,
    risk_level: str = 'medium'
)
```

**Parameters:**
- `use_topology` (bool): Use topology-aware thresholds. Default: True
- `use_multiscale` (bool): Use multi-scale regime detection. Default: True
- `risk_level` (str): Risk tolerance ('low', 'medium', 'high'). Default: 'medium'

#### Methods

##### `fit()`

Initialize system with historical data.
```python
def fit(
    self,
    prices: pd.Series,
    volumes: pd.Series,
    returns: Optional[pd.Series] = None
) -> 'TradingChannelSystem'
```

**Parameters:**
- `prices` (pd.Series): Historical prices
- `volumes` (pd.Series): Historical volumes
- `returns` (pd.Series, optional): Historical returns (for supervised learning)

**Returns:**
- `self`: For method chaining

**Example:**
```python
import pandas as pd
from channelpy.applications import TradingChannelSystem

# Load data
df = pd.read_csv('historical_data.csv', parse_dates=['Date'])
df = df.set_index('Date')

# Create and fit system
system = TradingChannelSystem(risk_level='medium')
system.fit(
    prices=df['Close'],
    volumes=df['Volume'],
    returns=df['Returns']
)

print("System initialized and calibrated")
```

##### `process_tick()`

Process new market data and generate signal.
```python
def process_tick(
    self,
    price: float,
    volume: float,
    timestamp: Optional[pd.Timestamp] = None
) -> Dict[str, Any]
```

**Parameters:**
- `price` (float): Current price
- `volume` (float): Current volume
- `timestamp` (pd.Timestamp, optional): Timestamp

**Returns:**
- `dict`: Signal dictionary with keys:
  - `'action'` (str): 'BUY', 'SELL', or 'HOLD'
  - `'confidence'` (float): Confidence level [0, 1]
  - `'position_size'` (float): Suggested position size [0, 1]
  - `'reason'` (str): Human-readable explanation
  - `'states'` (dict): Channel states for each feature
  - `'regime'` (str): Current market regime

**Example:**
```python
# Process real-time tick
signal = system.process_tick(
    price=152.50,
    volume=1200000,
    timestamp=pd.Timestamp.now()
)

# Execute based on signal
if signal['action'] == 'BUY' and signal['confidence'] > 0.7:
    position_size = signal['position_size'] * account_size
    execute_buy(position_size)
    print(f"BUY signal: {signal['reason']}")

elif signal['action'] == 'SELL' and signal['confidence'] > 0.7:
    execute_sell()
    print(f"SELL signal: {signal['reason']}")
```

##### `backtest()`

Backtest system on historical data.
```python
def backtest(
    self,
    prices: pd.Series,
    volumes: pd.Series,
    initial_capital: float = 10000.0
) -> Dict[str, Any]
```

**Parameters:**
- `prices` (pd.Series): Historical prices
- `volumes` (pd.Series): Historical volumes
- `initial_capital` (float): Starting capital. Default: 10000.0

**Returns:**
- `dict`: Backtest results with keys:
  - `'total_return'` (float): Total return (%)
  - `'sharpe_ratio'` (float): Risk-adjusted return
  - `'max_drawdown'` (float): Maximum drawdown (%)
  - `'win_rate'` (float): Fraction of winning trades
  - `'num_trades'` (int): Total number of trades
  - `'equity_curve'` (pd.Series): Equity over time

**Example:**
```python
# Backtest on historical data
results = system.backtest(
    prices=test_df['Close'],
    volumes=test_df['Volume'],
    initial_capital=10000.0
)

print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
print(f"Win Rate: {results['win_rate']:.2f}%")
print(f"Number of Trades: {results['num_trades']}")

# Plot equity curve
results['equity_curve'].plot(title='Backtest Equity Curve')
```

##### `get_current_regime()`

Get current market regime.
```python
def get_current_regime(self) -> Dict[str, Any]
```

**Returns:**
- `dict`: Regime information:
  - `'regime'` (str): Regime type ('stable', 'volatile', 'trending', etc.)
  - `'confidence'` (float): Detection confidence
  - `'characteristics'` (dict): Regime characteristics

---

### `TechnicalIndicatorEncoder`

Encode technical indicators to channel states.
```python
class TechnicalIndicatorEncoder:
    """
    Encode technical analysis indicators
    
    Supports:
    - Moving averages (SMA, EMA)
    - Momentum indicators (RSI, MACD)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators
    
    Examples
    --------
    >>> encoder = TechnicalIndicatorEncoder()
    >>> 
    >>> # Encode RSI
    >>> rsi_state = encoder.encode_rsi(rsi_value=45)
    >>> 
    >>> # Encode MACD
    >>> macd_state = encoder.encode_macd(
    ...     macd_line=0.5,
    ...     signal_line=0.3
    ... )
    """
```

#### Methods

##### `encode_rsi()`
```python
def encode_rsi(
    self,
    rsi_value: float,
    oversold: float = 30,
    overbought: float = 70
) -> State
```

**Parameters:**
- `rsi_value` (float): RSI value [0, 100]
- `oversold` (float): Oversold threshold. Default: 30
- `overbought` (float): Overbought threshold. Default: 70

**Returns:**
- `State`: Encoded state
  - ψ (PSI): Overbought (RSI > 70)
  - δ (DELTA): Moderate high (50 < RSI < 70)
  - φ (PHI): Moderate low (30 < RSI < 50)
  - ∅ (EMPTY): Oversold (RSI < 30)

##### `encode_macd()`
```python
def encode_macd(
    self,
    macd_line: float,
    signal_line: float
) -> State
```

**Parameters:**
- `macd_line` (float): MACD line value
- `signal_line` (float): Signal line value

**Returns:**
- `State`: Encoded state based on crossover

##### `encode_bollinger_bands()`
```python
def encode_bollinger_bands(
    self,
    price: float,
    middle_band: float,
    upper_band: float,
    lower_band: float
) -> State
```

**Parameters:**
- `price` (float): Current price
- `middle_band` (float): Middle Bollinger Band (SMA)
- `upper_band` (float): Upper band
- `lower_band` (float): Lower band

**Returns:**
- `State`: Position relative to bands

---

### `TradingSignalInterpreter`

Interpret channel states to trading signals.
```python
class TradingSignalInterpreter:
    """
    Convert channel states to actionable trading signals
    
    Uses pattern matching on state combinations to generate
    buy/sell/hold decisions with confidence levels.
    
    Examples
    --------
    >>> interpreter = TradingSignalInterpreter()
    >>> 
    >>> states = {
    ...     'price': PSI,
    ...     'volume': PSI,
    ...     'volatility': PHI
    ... }
    >>> 
    >>> signal = interpreter.interpret(states)
    >>> print(signal['action'])  # 'BUY'
    """
```

#### Methods

##### `interpret()`
```python
def interpret(
    self,
    states: Dict[str, State],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `states` (Dict[str, State]): Channel states for each feature
- `context` (Dict, optional): Additional context (regime, time, etc.)

**Returns:**
- `dict`: Trading signal

---

## Medical Applications

### `MedicalDiagnosisSystem`

Medical diagnosis using channel algebra.
```python
class MedicalDiagnosisSystem:
    """
    Medical diagnosis system with explainable reasoning
    
    Features:
    - Multi-symptom encoding
    - Test result integration
    - Hierarchical diagnosis (organ → disease → subtype)
    - Confidence scoring
    - Explanation generation
    
    Examples
    --------
    >>> from channelpy.applications import MedicalDiagnosisSystem
    >>> 
    >>> system = MedicalDiagnosisSystem(specialty='cardiology')
    >>> 
    >>> # Input patient data
    >>> patient_data = {
    ...     'symptoms': {
    ...         'chest_pain': 'severe',
    ...         'shortness_of_breath': 'moderate',
    ...         'fatigue': 'mild'
    ...     },
    ...     'test_results': {
    ...         'ecg': 'abnormal',
    ...         'troponin': 2.5,
    ...         'creatinine': 1.1
    ...     },
    ...     'vital_signs': {
    ...         'blood_pressure': '140/90',
    ...         'heart_rate': 95
    ...     }
    ... }
    >>> 
    >>> # Get diagnosis
    >>> diagnosis = system.diagnose(patient_data)
    >>> print(diagnosis['primary_diagnosis'])
    >>> print(f"Confidence: {diagnosis['confidence']:.1%}")
    >>> print(f"Reasoning: {diagnosis['explanation']}")
    """
```

#### Constructor
```python
def __init__(
    self,
    specialty: str = 'general',
    confidence_threshold: float = 0.6
)
```

**Parameters:**
- `specialty` (str): Medical specialty ('general', 'cardiology', 'neurology', etc.)
- `confidence_threshold` (float): Minimum confidence for positive diagnosis

#### Methods

##### `diagnose()`
```python
def diagnose(
    self,
    patient_data: Dict[str, Any]
) -> Dict[str, Any]
```

**Parameters:**
- `patient_data` (dict): Patient information including:
  - `'symptoms'`: Symptom descriptions
  - `'test_results'`: Lab/diagnostic test results
  - `'vital_signs'`: Vital sign measurements
  - `'history'`: Medical history (optional)

**Returns:**
- `dict`: Diagnosis with keys:
  - `'primary_diagnosis'` (str): Most likely diagnosis
  - `'differential'` (List[str]): Alternative diagnoses
  - `'confidence'` (float): Confidence level [0, 1]
  - `'explanation'` (str): Reasoning chain
  - `'recommended_tests'` (List[str]): Suggested additional tests
  - `'states'` (dict): Channel states for each input

**Example:**
```python
from channelpy.applications import MedicalDiagnosisSystem

system = MedicalDiagnosisSystem(specialty='cardiology')

patient_data = {
    'symptoms': {
        'chest_pain': 'severe',
        'shortness_of_breath': 'moderate'
    },
    'test_results': {
        'troponin': 2.5,  # Elevated
        'ecg': 'ST_elevation'
    },
    'vital_signs': {
        'blood_pressure': '140/90',
        'heart_rate': 95
    }
}

diagnosis = system.diagnose(patient_data)

print(f"Diagnosis: {diagnosis['primary_diagnosis']}")
print(f"Confidence: {diagnosis['confidence']:.1%}")
print(f"\nReasoning:")
print(diagnosis['explanation'])

if diagnosis['recommended_tests']:
    print(f"\nRecommended tests:")
    for test in diagnosis['recommended_tests']:
        print(f"  - {test}")

# Differential diagnoses
if diagnosis['differential']:
    print(f"\nDifferential diagnoses:")
    for alt in diagnosis['differential']:
        print(f"  - {alt}")
```

##### `encode_symptoms()`
```python
def encode_symptoms(
    self,
    symptoms: Dict[str, str]
) -> Dict[str, State]
```

**Parameters:**
- `symptoms` (Dict[str, str]): Symptom severity mapping

**Returns:**
- `Dict[str, State]`: Encoded channel states

##### `encode_test_results()`
```python
def encode_test_results(
    self,
    test_results: Dict[str, Any]
) -> Dict[str, State]
```

**Parameters:**
- `test_results` (Dict[str, Any]): Test results

**Returns:**
- `Dict[str, State]`: Encoded states

---

### `SymptomEncoder`

Encode symptoms to channel states.
```python
class SymptomEncoder:
    """
    Encode symptom descriptions
    
    Maps symptom severity to states:
    - ψ (PSI): Severe symptom
    - δ (DELTA): Moderate symptom
    - φ (PHI): Mild symptom
    - ∅ (EMPTY): Symptom absent
    
    Examples
    --------
    >>> encoder = SymptomEncoder()
    >>> state = encoder.encode('chest_pain', severity='severe')
    >>> print(state)  # ψ
    """
```

---

### `TestResultEncoder`

Encode medical test results.
```python
class TestResultEncoder:
    """
    Encode laboratory and diagnostic test results
    
    Handles:
    - Continuous values (with normal ranges)
    - Categorical results
    - Binary results (positive/negative)
    
    Examples
    --------
    >>> encoder = TestResultEncoder()
    >>> 
    >>> # Encode continuous value
    >>> troponin_state = encoder.encode_continuous(
    ...     test_name='troponin',
    ...     value=2.5,
    ...     normal_range=(0, 0.4)
    ... )
    >>> 
    >>> # Encode categorical
    >>> ecg_state = encoder.encode_categorical(
    ...     test_name='ecg',
    ...     value='ST_elevation',
    ...     severity_map={'normal': 0, 'abnormal': 1, 'ST_elevation': 3}
    ... )
    """
```

---

## Signal Processing Applications

### `SignalChannelSystem`

Signal analysis using channel algebra.
```python
class SignalChannelSystem:
    """
    Analyze time-series signals
    
    Applications:
    - Anomaly detection
    - Quality assessment
    - Pattern recognition
    - Change point detection
    
    Examples
    --------
    >>> from channelpy.applications import SignalChannelSystem
    >>> 
    >>> system = SignalChannelSystem(
    ...     signal_type='sensor',
    ...     sampling_rate=100.0
    ... )
    >>> 
    >>> # Process signal
    >>> for sample in signal_data:
    ...     result = system.process_sample(sample)
    ...     
    ...     if result['anomaly_detected']:
    ...         print(f"Anomaly at t={result['timestamp']}")
    ...         print(f"Confidence: {result['confidence']:.2f}")
    """
```

#### Constructor
```python
def __init__(
    self,
    signal_type: str = 'sensor',
    sampling_rate: float = 100.0,
    window_size: int = 1000
)
```

**Parameters:**
- `signal_type` (str): Type of signal ('sensor', 'audio', 'ecg', etc.)
- `sampling_rate` (float): Samples per second
- `window_size` (int): Analysis window size

#### Methods

##### `process_sample()`
```python
def process_sample(
    self,
    value: float,
    timestamp: Optional[float] = None
) -> Dict[str, Any]
```

**Parameters:**
- `value` (float): Signal sample value
- `timestamp` (float, optional): Sample timestamp

**Returns:**
- `dict`: Analysis result with keys:
  - `'anomaly_detected'` (bool): Whether anomaly found
  - `'confidence'` (float): Detection confidence
  - `'signal_quality'` (str): Quality assessment
  - `'state'` (State): Channel state
  - `'features'` (dict): Extracted features

##### `detect_change_points()`
```python
def detect_change_points(
    self,
    signal: np.ndarray
) -> List[int]
```

**Parameters:**
- `signal` (np.ndarray): Signal data

**Returns:**
- `List[int]`: Indices of detected change points

##### `assess_quality()`
```python
def assess_quality(
    self,
    signal: np.ndarray
) -> Dict[str, Any]
```

**Parameters:**
- `signal` (np.ndarray): Signal data

**Returns:**
- `dict`: Quality metrics

---

## Application Template

### `ApplicationTemplate`

Base class for creating custom applications.
```python
class ApplicationTemplate:
    """
    Template for building domain-specific applications
    
    Provides structure for:
    - Data preprocessing
    - Feature extraction
    - Encoding
    - Interpretation
    - Decision making
    
    Subclass this to create new applications.
    
    Examples
    --------
    >>> class MyApplication(ApplicationTemplate):
    ...     def preprocess(self, data):
    ...         # Custom preprocessing
    ...         return processed_data
    ...     
    ...     def extract_features(self, data):
    ...         # Custom feature extraction
    ...         return features
    ...     
    ...     def interpret_states(self, states):
    ...         # Custom interpretation
    ...         return decision
    """
```

#### Methods to Override

##### `preprocess()`
```python
def preprocess(self, data: Any) -> Any:
    """Preprocess raw data"""
    pass
```

##### `extract_features()`
```python
def extract_features(self, data: Any) -> Dict[str, float]:
    """Extract features from preprocessed data"""
    pass
```

##### `encode_features()`
```python
def encode_features(self, features: Dict[str, float]) -> Dict[str, State]:
    """Encode features to channel states"""
    pass
```

##### `interpret_states()`
```python
def interpret_states(
    self,
    states: Dict[str, State],
    context: Optional[Dict] = None
) -> Dict[str, Any]:
    """Interpret states to make decisions"""
    pass
```

---

## Utility Functions

### `create_application()`

Factory function for creating applications.
```python
def create_application(
    domain: str,
    **kwargs
) -> ApplicationTemplate
```

**Parameters:**
- `domain` (str): Application domain ('trading', 'medical', 'signal', etc.)
- `**kwargs`: Domain-specific parameters

**Returns:**
- `ApplicationTemplate`: Configured application instance

**Example:**
```python
from channelpy.applications import create_application

# Create trading application
trading_app = create_application(
    domain='trading',
    risk_level='medium',
    use_topology=True
)

# Create medical application
medical_app = create_application(
    domain='medical',
    specialty='cardiology',
    confidence_threshold=0.7
)
```

---

## Complete Example: Multi-Domain System
```python
import pandas as pd
from channelpy.applications import (
    TradingChannelSystem,
    MedicalDiagnosisSystem,
    SignalChannelSystem
)

# Trading Example
print("=== Trading Application ===")
trading_system = TradingChannelSystem(risk_level='medium')

# Load and process data
df = pd.read_csv('market_data.csv')
trading_system.fit(df['Close'], df['Volume'])

signal = trading_system.process_tick(
    price=150.0,
    volume=1000000
)
print(f"Action: {signal['action']}")
print(f"Confidence: {signal['confidence']:.2f}")
print(f"Reason: {signal['reason']}")

# Medical Example
print("\n=== Medical Application ===")
medical_system = MedicalDiagnosisSystem(specialty='cardiology')

patient_data = {
    'symptoms': {'chest_pain': 'severe'},
    'test_results': {'troponin': 2.5},
    'vital_signs': {'blood_pressure': '140/90'}
}

diagnosis = medical_system.diagnose(patient_data)
print(f"Diagnosis: {diagnosis['primary_diagnosis']}")
print(f"Confidence: {diagnosis['confidence']:.1%}")

# Signal Processing Example
print("\n=== Signal Processing Application ===")
signal_system = SignalChannelSystem(signal_type='sensor')

for sample in sensor_data:
    result = signal_system.process_sample(sample)
    if result['anomaly_detected']:
        print(f"Anomaly detected! Confidence: {result['confidence']:.2f}")
```

---

## Best Practices

### 1. Application Design

- **Start simple**: Begin with basic encoding rules
- **Add complexity gradually**: Introduce topology-awareness as needed
- **Validate thoroughly**: Test on diverse data before deployment
- **Document decisions**: Explain why specific encoding/interpretation rules were chosen

### 2. Feature Selection

- **Choose informative features**: Features that discriminate between outcomes
- **Avoid redundancy**: Don't encode highly correlated features separately
- **Consider temporal dynamics**: For time-series, include momentum/change features

### 3. Threshold Setting

- **Use adaptive thresholds**: Let data topology guide threshold selection
- **Validate on hold-out data**: Ensure thresholds generalize
- **Monitor drift**: Track threshold evolution over time

### 4. Interpretation

- **Multiple evidence sources**: Combine multiple feature states
- **Context matters**: Use regime/environment information
- **Confidence scoring**: Always report confidence levels
- **Explain decisions**: Provide human-readable reasoning

---

## See Also

- [Trading Tutorial](../tutorials/04_trading_bot.md) - Complete trading example
- [Medical Tutorial](../tutorials/05_medical_diagnosis.md) - Complete medical example
- [Custom Encoder Guide](../how_to_guides/custom_encoder.md) - Building custom encoders
- [Pipeline API](pipeline.md) - Pipeline building blocks
- [Adaptive API](adaptive.md) - Adaptive threshold systems