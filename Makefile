# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

# Define the expected virtual environment path
VENV_DIR := .venv

.DEFAULT_GOAL := default

# Declare phony targets to avoid conflicts with files
.PHONY: check-venv qa build update all qa

default: qa

# Check if the correct virtual environment is active
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ No virtual environment is active. Please activate the virtual environment by running 'source ./setup.sh'."; \
		exit 1; \
	fi
	@if [ "$$VIRTUAL_ENV" != "$(PWD)/$(VENV_DIR)" ]; then \
		echo "❌ Wrong virtual environment is active ($$VIRTUAL_ENV). Expected $(PWD)/$(VENV_DIR). Please deactivate the current one with 'deactivate' and run 'source ./setup.sh'."; \
		exit 1; \
	fi

	@uv lock --locked

	@echo "✅ Correct virtual environment is active: $$VIRTUAL_ENV"

# Run quality assurance checks
qa: check-venv
	@echo "🔍 Running quality assurance checks..."
	@git add . || { echo "❌ Failed to stage changes."; exit 1; }
	@pre-commit run --all-files || { echo "❌ Quality assurance checks failed."; exit 1; }
	@echo "✅ Quality assurance checks complete!"

# Build the package
build: check-venv
	@echo "🔨 Building the project..."
	@uv build || { echo "❌ Build failed."; exit 1; }
	@echo "✅ Build complete!"


# Update dependencies and pre-commit hooks
update: check-venv
	@echo "🔄 Updating dependencies and pre-commit hooks..."
	@uv lock --upgrade || { echo "❌ Failed to upgrade uv lock."; exit 1; }
	@uv sync --extra optional  || { echo "❌ Failed to sync uv."; exit 1; }
	@pre-commit autoupdate || { echo "❌ Failed to update pre-commit hooks."; exit 1; }
	@echo "✅ Update complete!"

unused-packages:
	@echo "🔍 Detecting unused packages..."
	@deptry src

test:
	@echo "🔍 Running tests..."
	@pytest -v --tb=short --disable-warnings --maxfail=1 || { echo "❌ Tests failed."; exit 1; }
	@echo "✅ All tests passed!"
