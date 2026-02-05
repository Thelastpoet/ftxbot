# Contributing to Forex Trading Bot

Thank you for your interest in contributing to the Forex Trading Bot! This document outlines the guidelines for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Include your environment details (OS, Python version, MT5 version)
- Provide any relevant logs or error messages

### Suggesting Enhancements

- Check if the enhancement has already been suggested
- Use a clear and descriptive title
- Provide a detailed explanation of the proposed enhancement
- Explain why this enhancement would be useful

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Ensure your code follows the existing style
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to your fork (`git push origin feature/amazing-feature`)
9. Create a pull request

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to functions and classes
- Keep functions reasonably short and focused
- Write clear, concise comments when necessary

## Testing

While this project doesn't currently have an automated test suite, please manually test your changes before submitting a pull request. Consider adding tests for new functionality.

## Documentation

- Update documentation when adding new features
- Keep README.md up to date with any changes
- Comment your code appropriately

## Project Structure

- `main.py`: Main trading loop and bot orchestration
- `strategy.py`: Trading strategy implementation
- `risk_manager.py`: Risk management logic
- `market_data.py`: Market data handling
- `trailing_stop.py`: Trailing stop logic
- `mt5_client.py`: MT5 platform interface
- `state_manager.py`: State persistence
- `trade_logger.py`: Trade logging
- `utils.py`: Utility functions
- `logging_utils.py`: Logging configuration

## Questions?

If you have any questions about contributing, feel free to open an issue for discussion.