#!/bin/bash
# filepath: setup.sh
# Author: Angel Martinez-Tenor, 2025
# Description: Sets up a Python project using pyproject.toml, managing a virtual environment, and syncing dependencies with uv.
# Usage: source ./setup.sh

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
        error) echo -e "${RED}❌ $2${NC}" >&2; return 1 ;;
        info) echo -e "${GREEN}ℹ️ $2${NC}" ;;
    esac
}

check_cmd() { command -v "$1" &>/dev/null; }

main() {
    log info "Starting project setup..."

    # Check prerequisites
    check_cmd uv || { log error "'uv' not installed. Install with 'pipx install uv'."; return 1; }
    [[ -f "pyproject.toml" ]] || { log error "'pyproject.toml' not found."; return 1; }

    # Extract project name
    local project_name
    project_name=$(grep -m 1 "name" pyproject.toml | sed -E 's/.*name = "([^"]+)".*/\1/' | tr -d '[:space:]') || project_name="project"
    [[ "$project_name" == "project" ]] && log warn "Project name not found in pyproject.toml. Using 'project'."
    log success "Project name: '$project_name'."

    # Manage virtual environment
    local venv_dir=".venv"
    if [[ ! -d "$venv_dir" || ! -f "$venv_dir/bin/activate" ]]; then
        log info "Creating virtual environment..."
        uv venv "$venv_dir" --prompt "$project_name" >&2 || { log error "Failed to create virtual environment."; return 1; }
        log success "Virtual environment created."
    else
        log success "Virtual environment exists."
    fi

    # Activate virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log info "Activating virtual environment..."
        source "$venv_dir/bin/activate" || { log error "Failed to activate virtual environment."; return 1; }
        check_cmd python || { log error "Python not found in virtual environment."; return 1; }
        log success "Virtual environment activated."
    elif [[ "${VIRTUAL_ENV:-}" != "$(pwd)/$venv_dir" ]]; then
        log warn "Another virtual environment is active (${VIRTUAL_ENV:-}). Run 'deactivate'."
        return 1
    else
        log success "Virtual environment already active."
    fi

    # Sync dependencies
    log info "Syncing dependencies..."
    uv sync --extra dev >&2 || { log error "Failed to sync dependencies."; return 1; }
    log success "Dependencies synced."

    log success "Project '$project_name' setup completed."
}

main
return 0
