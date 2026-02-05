# Architecture

This document describes the architecture and design of the Forex Trading Bot.

## System Overview

The Forex Trading Bot is a modular system built around the MetaTrader 5 platform. It follows a component-based architecture where each module handles specific responsibilities.

## Core Components

### Main Controller (`main.py`)
The main controller orchestrates the entire system:
- Manages the main trading loop
- Coordinates between components
- Handles initialization and shutdown
- Processes trading signals
- Manages open positions

### Trading Strategy (`strategy.py`)
Implements the trading logic:
- Analyzes market data
- Identifies swing points
- Detects breakouts
- Generates trading signals
- Calculates entry/exit levels

### Risk Manager (`risk_manager.py`)
Handles all risk-related functions:
- Calculates position sizes
- Validates trade parameters
- Checks risk limits
- Manages account protection

### Market Data Handler (`market_data.py`)
Manages market data retrieval:
- Fetches historical data from MT5
- Processes and formats data
- Provides data to strategy components

### MT5 Client (`mt5_client.py`)
Interface to the MetaTrader 5 platform:
- Handles all MT5 API calls
- Manages connections
- Executes trades
- Retrieves account information

### Trailing Stops (`trailing_stop.py`)
Manages trailing stop logic:
- Updates stop losses based on price action
- Implements R-multiple logic
- Handles position management

### State Manager (`state_manager.py`)
Maintains persistent state:
- Prevents duplicate signals
- Saves/restores state across restarts
- Manages signal tracking

### Trade Logger (`trade_logger.py`)
Records all trading activity:
- Logs all trades
- Maintains trade history
- Provides trade analytics

## Data Flow

### Signal Generation Process
1. Market data handler fetches required data
2. Strategy analyzes data and detects breakouts
3. Risk manager validates trade parameters
4. MT5 client executes the trade
5. Trade logger records the transaction
6. Trailing stops manage the position

### Position Monitoring Process
1. Main controller polls for open positions
2. Trailing stops update stop losses if needed
3. Risk manager monitors account health
4. Trade logger updates position status

## Configuration System

### Config Class (`main.py`)
Centralizes all configuration:
- Loads settings from JSON file
- Handles command-line overrides
- Provides configuration to all components

### Configuration Structure
- Trading settings
- Risk management parameters
- Symbol-specific configurations
- Runtime parameters

## Error Handling

### Connection Management
- Automatic reconnection to MT5
- Retry mechanisms for failed operations
- Graceful degradation when services unavailable

### Trade Execution
- Validation before execution
- Error handling during execution
- Recovery from partial failures

## Threading and Concurrency

### Async Operations
- Non-blocking data retrieval
- Concurrent symbol processing
- Background monitoring tasks

### Thread Safety
- Proper locking for shared resources
- Safe state management
- Protected configuration access

## Extensibility

### Plugin Architecture
Components are designed to be replaceable:
- Strategy implementations can be swapped
- Risk management logic can be customized
- Data sources can be modified

### Event System
Components communicate through:
- Direct method calls
- Shared state management
- Event-based notifications

## Security Considerations

### Input Validation
- All configuration inputs are validated
- Trade parameters are checked
- Market data is verified

### Access Control
- MT5 credentials are handled securely
- Trade execution is limited to authorized operations
- Account information is protected

## Performance Considerations

### Efficiency
- Minimal data transfer
- Efficient data structures
- Optimized algorithms

### Scalability
- Support for multiple symbols
- Configurable polling intervals
- Resource management

## Integration Points

### MetaTrader 5
- Uses MT5 Python API
- Leverages MT5's execution engine
- Integrates with MT5's risk management

### External Systems
- File-based logging
- JSON configuration
- Standard Python libraries

## Design Patterns

### Singleton Pattern
- MT5 client is shared across components
- Configuration is centralized

### Observer Pattern
- Components react to market changes
- Position updates trigger trailing stops

### Factory Pattern
- Strategy creation is centralized
- Component initialization is standardized