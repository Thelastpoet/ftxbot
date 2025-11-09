# Repository Guidelines

## Project Structure & Module Organization
- `main.py` — entrypoint; CLI, event loop, logging, config load.
- `strategy.py` — signal generation (e.g., `PurePriceActionStrategy`).
- `risk_manager.py` — position sizing, SL/TP, and guardrails.
- `market_data.py` / `mt5_client.py` — data access and broker bridge.
- `patterns.py` — pattern scoring helpers used by strategies.
- `trade_logger.py` — trade/decision logging utilities.
- `utils.py` — shared helpers (e.g., pip size resolution).
- `config.json` — trading/risk settings and per-symbol overrides.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv` then activate (`.\.venv\Scripts\Activate` on PowerShell).
- Install deps: `pip install -r requirements.txt`.
- Run locally: `python main.py --help` (then run with desired args).
- Optional tooling: `pip install black flake8 pytest`.
  - Format: `black .`  •  Lint: `flake8`  •  Tests: `pytest -q`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indent; prefer type hints and docstrings.
- Naming: modules/functions `lower_snake_case`; classes `CamelCase`; constants `UPPER_SNAKE_CASE`.
- Keep I/O and orchestration in `main.py`; keep modules pure/testable.
- Log with module‑scoped loggers; avoid prints in library code.

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/` named `test_*.py` (e.g., `tests/test_strategy.py`).
- Focus on `strategy.py` and `risk_manager.py` deterministically; stub broker/data layers.
- Aim for meaningful coverage on decision logic; keep fixtures small and explicit.

## Commit & Pull Request Guidelines
- History is informal; please adopt concise, imperative subjects. Prefer Conventional Commits (e.g., `feat: add breakout filter`, `fix: pip size for XAUUSD`).
- PRs should include: purpose/summary, scope of changes, testing notes (logs or screenshots if relevant), and any config changes (`config.json` diffs).

## Security & Configuration Tips
- `config.json` holds trading/risk settings; avoid storing secrets. If credentials are ever needed, prefer environment variables and document them in the PR.
- For local overrides, consider an untracked `config.local.json` and merge in `main.py` if added.

## Agent-Specific Instructions
- Scope: this file applies to the entire repo.
- Make minimal, focused changes; do not reformat unrelated code or add dependencies without need.
- Keep public behavior stable; update docs alongside code changes.

