# Default to current directory if DIR is not provided
DIR ?= .

.PHONY: black format check lint help

black:
	black $(DIR)

format: ## Auto-format code using black and isort
	isort $(DIR)
	black $(DIR)

check: ## Check formatting without modifying files (useful for CI)
	isort --check-only $(DIR)
	black --check $(DIR)

lint: ## Run flake8
	flake8 $(DIR)

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'