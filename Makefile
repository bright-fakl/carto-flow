.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry src

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

# mkdocs-gallery 0.10.4 (latest) imports mkdocs.utils.warning_filter, removed in
# mkdocs 1.2, and jupyter_core warns unless JUPYTER_PLATFORM_DIRS is set. Both
# fire at import time, before mkdocs loads docs/hooks/quiet_third_party.py.
DOCS_ENV = JUPYTER_PLATFORM_DIRS=1 \
	PYTHONWARNINGS='ignore:warning_filter doesn.t do anything:DeprecationWarning'

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@$(DOCS_ENV) uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@$(DOCS_ENV) uv run mkdocs serve

.PHONY: bump-version
bump-version: ## Bump version using script (e.g., make bump-version VERSION=1.0.0)
	@if [ -z "${VERSION}" ]; then \
		echo "❌ Error: VERSION argument is required (e.g., make bump-version VERSION=1.0.0)"; \
		exit 1; \
	fi
	@echo "🚀 Bumping version to ${VERSION}"
	@uv run python scripts/bump_version.py ${VERSION}

.PHONY: create-tag
create-tag: ## Create and push git tag (e.g., make create-tag VERSION=1.0.0)
	@if [ -z "${VERSION}" ]; then \
		echo "❌ Error: VERSION argument is required (e.g., make create-tag VERSION=1.0.0)"; \
		exit 1; \
	fi
	@echo "🚀 Creating tag ${VERSION}"
	@git tag ${VERSION}
	@git push origin ${VERSION}
	@echo "✅ Tag ${VERSION} created and pushed"

.PHONY: release
release: bump-version create-tag ## Bump version and create tag (e.g., make release VERSION=1.0.0)

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
