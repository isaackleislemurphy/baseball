# Default to current directory if DIR is not provided
DIR ?= .

.PHONY: black

black:
	black $(DIR)

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

format: ## Auto-format code using black and isort
	isort $(SRC)
	black $(SRC)

check: ## Check formatting without modifying files (useful for CI)
	isort --check-only $(SRC)
	black --check $(SRC)

lint: ## Run flake8 (or other linters)
	flake8 $(SRC)