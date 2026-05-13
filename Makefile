.DEFAULT_GOAL := help

.PHONY: help lock setup-project test test-coverage build

help:  ## Show this help message
	@echo "parquery — A query and aggregation framework for Parquet"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

lock:  ## Update uv.lock against CodeArtifact (requires aws sso login)
	@echo "Updating uv.lock..."
	$(eval CODEARTIFACT_TOKEN := $(shell aws codeartifact get-authorization-token --domain visualfabriq --query authorizationToken --output text))
	UV_EXTRA_INDEX_URL="https://aws:$(CODEARTIFACT_TOKEN)@visualfabriq-103613169272.d.codeartifact.eu-west-1.amazonaws.com/pypi/private/simple/" uv lock
	@echo "Lockfile updated. Commit uv.lock."

setup-project:  ## Sync dependencies (test group) using the committed lockfile
	uv sync --group test --frozen

test:  ## Run the pytest test suite
	uv run pytest tests

test-coverage:  ## Run tests with coverage reports (XML + HTML)
	uv run coverage run -m pytest tests --junitxml=test-results/junit.xml
	uv run coverage xml -o coverage.xml
	uv run coverage html -d coverage-html

build:  ## Build wheel and sdist with uv
	uv build
