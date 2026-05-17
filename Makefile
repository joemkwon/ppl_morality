.PHONY: install dev test lint fmt repro figs clean check-no-pii

install:
	pip install -e .

dev:
	pip install -e ".[dev,llm]"

test:
	pytest

lint:
	ruff check src tests experiments reproduce.py
	black --check src tests experiments reproduce.py

fmt:
	ruff check --fix src tests experiments reproduce.py
	black src tests experiments reproduce.py

# Full paper reproduction (Fig 1, 4, 6, 7). Deterministic with --seed 0.
repro:
	python reproduce.py --seed 0

# Fast smoke run (coarse bins) to sanity-check the pipeline end-to-end.
figs:
	python reproduce.py --seed 0 --fast

# Guard: fail if any participant identifiers or secrets would be shipped.
check-no-pii:
	@! grep -rIl -E "prolific_id|user_agent|\\bsk-[A-Za-z0-9]{20}|OPENAI_API_KEY *= *['\"]" \
		--include="*.csv" --include="*.py" --include="*.json" data src experiments LLM \
		2>/dev/null || (echo "POTENTIAL PII / SECRET FOUND" && exit 1)
	@echo "No PII or hardcoded secrets detected in shipped paths."

clean:
	rm -rf figures/*.png results forward_model_table*.json
	find . -name __pycache__ -type d -exec rm -rf {} +
