.PHONY: help install test lint clean build run-live run-backtest docker-up docker-down

PYTHON := python3
PIP := pip3
CARGO := cargo

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: install-python install-rust ## Install all dependencies

install-python: ## Install Python dependencies
	cd ml && $(PIP) install -r requirements.txt && $(PIP) install -e .

install-rust: ## Build the Rust proxy
	cd proxy && $(CARGO) build --release

test: ## Run the test suite
	cd ml && $(PYTHON) -m pytest tests/ -v --cov=flash_crash_watchdog

lint: ## Lint Python code
	cd ml && $(PYTHON) -m flake8 flash_crash_watchdog/ tests/

run-live: ## Run the live detector against Binance WebSocket
	cd ml && $(PYTHON) -m flash_crash_watchdog.cli live --symbol BTCUSDT

run-backtest: ## Run the backtest on sample data
	cd ml && $(PYTHON) -m flash_crash_watchdog.cli backtest --data ../data/sample.parquet

download-data: ## Download sample Binance data (May 19, 2021 BTC crash)
	cd ml && $(PYTHON) -m flash_crash_watchdog.data.download_binance --symbol BTCUSDT --date 2021-05-19 --out ../data/

train-tcn: ## Train the TCN model on FI-2010
	cd ml && $(PYTHON) -m flash_crash_watchdog.cli train --data ../data/fi2010/ --model ../configs/tcn_baseline.yml

dashboard: ## Start the Next.js dashboard
	cd dashboard && npm install && npm run dev

docker-up: ## Start all services via Docker Compose
	docker-compose up -d

docker-down: ## Stop all Docker services
	docker-compose down

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf ml/*.egg-info ml/build ml/dist
	cd proxy && $(CARGO) clean

build: ## Build everything (Python wheel + Rust binary)
	cd ml && $(PYTHON) setup.py bdist_wheel
	cd proxy && $(CARGO) build --release
