#!/bin/bash

# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

# Description: Sets up a Python project using pyproject.toml, managing a virtual env, and syncing dependencies with uv.
# Usage: source ./setup.sh [--reset, --noqa]


# Check if sourced
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && { echo -e "\e[31m❌ Script must be sourced: 'source $0'.\e[0m" >&2; exit 1; }

# Colors
GREEN='\e[32m'
YELLOW='\e[33m'
RED='\e[31m'
NC='\e[0m'

# Logging
log() {
    case $1 in
        success) echo -e "${GREEN}✅ $2${NC}" ;;
        warn) echo -e "${YELLOW}⚠️ $2${NC}" ;;
        error) echo -e "${RED}❌ $2${NC}. Execute 'setup --reset' to remove the existing virtual environment." >&2; return 1 ;;
        info) echo -e "${GREEN}ℹ️ $2${NC}." ;;
    esac
}

check_cmd() { command -v "$1" &>/dev/null; }

main() {
    local reset=false
    [[ "$1" == "--reset" ]] && reset=true

    log info "Starting project setup..."

    # Deactivate any existing virtual environment
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        log info "Deactivating existing virtual environment (${VIRTUAL_ENV})..."
        deactivate || { log warning "Failed to deactivate existing virtual environment.";}
        log success "Existing virtual environment deactivated."
    fi

    # Check prerequisites
    check_cmd uv || { log error "'uv' not installed. Install with 'pipx install uv'."; return 1; }
    [[ -f "pyproject.toml" ]] || { log error "'pyproject.toml' not found."; return 1; }
    # check_cmd node || { log error "'node' not installed. Install with 'npm install -g nodejs'."; return 1; }

    # Extract project name
    local project_name
    project_name=$(grep -m 1 "name" pyproject.toml | sed -E 's/.*name = "([^"]+)".*/\1/' | tr -d '[:space:]') || project_name="project"
    [[ "$project_name" == "project" ]] && log warn "Project name not found in pyproject.toml. Using 'project'."
    log success "Project name: '$project_name'."

    # Manage virtual environment
    local venv_dir=".venv"
    if [[ "$reset" == true && -d "$venv_dir" ]]; then
        log info "Removing existing virtual environment..."
        rm -rf "$venv_dir" || { log error "Failed to remove existing virtual environment."; return 1; }
        log success "Existing virtual environment removed."
    fi

    if [[ ! -d "$venv_dir" || ! -f "$venv_dir/bin/activate" ]]; then
        log info "Creating virtual environment..."
        uv venv "$venv_dir" --prompt "$project_name" >&2 || { log error "Failed to create virtual environment."; return 1; }
        log success "Virtual environment created."
    else
        log success "Virtual environment exists."
    fi

    # Activate virtual environment
    log info "Activating virtual environment..."
    source "$venv_dir/bin/activate" || { log error "Failed to activate virtual environment."; return 1; }
    check_cmd python || { log error "Python not found in virtual environment."; return 1; }
    log success "Virtual environment activated."

    # Sync dependencies
    log info "Syncing dependencies..."
    uv sync --extra optional >&2 || { log error "Failed to sync dependencies."; return 1; }
    log success "Dependencies synced."

    log success "Project '$project_name' setup completed."
}

git flow init -d

main "$@"

if ! [[ " $* " =~ " --noqa " ]]; then
    make qa
else
    log info "Skipping QA as per --noqa flag."
fi

return 0
