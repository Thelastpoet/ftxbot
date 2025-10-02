# Repository Guidelines

## Project Structure & Module Organization
main.py drives the async trading loop, coordinating session control, execution, and shutdown. Strategy rules live in strategy.py, pricing utilities in market_data.py, platform access in mt5_client.py, and per-symbol state in symbol_runtime.py. isk_manager.py, optimizer.py, and calibrator.py store adaptive data under optimizer_states/ and calibrator_states/. Logs (orex_bot.log, 	rades.log) and ot_memory.json record diagnostics, while configurable inputs stay in config.json, symbol_configs/, and secrets in .env.

## Build, Test, and Development Commands
Create the virtualenv with python -m venv venv and activate using .\venv\Scripts\activate. Install required packages via pip install MetaTrader5 pandas numpy scipy (add TA-Lib if available). Launch the bot using python main.py --log-level INFO; pass alternative settings with --config path\to\config.demo.json or --symbol EURUSD. Reset optimizer baselines for clean experiments by removing files in optimizer_states\.

## Coding Style & Naming Conventions
Maintain PEP 8 formatting with four-space indentation, snake_case functions, and PascalCase classes. Type hints and dataclasses are expected for structured payloads; avoid implicit tuple returns. Prefer module-level logger = logging.getLogger(__name__) and keep log messages actionable.

## Testing Guidelines
No automated suite exists yet; validate behaviour against a demo MT5 account while running python main.py --log-level DEBUG. Review orex_bot.log and 	rades.log for order flow, and confirm state files update as expected. When touching calibration or optimizer code, clear the corresponding state directory and document observed win-rate or expectancy changes.

## Commit & Pull Request Guidelines
Recent history uses short, Title Case summaries (e.g., Tighten Risk Manager). Keep commits scoped and explain non-obvious effects in the body. Pull requests should link issues, describe strategy impact, list config edits, and include key log excerpts or MT5 screenshots; seek an extra review for execution or risk changes.

## Security & Configuration Tips
Do not commit live credentials; store them in .env and read through environment lookups. Exclude personal trade data from shared logs and keep .gitignore intact for log directories. When introducing new config keys, supply default values in the PR and alert operators so runtime copies stay aligned.
