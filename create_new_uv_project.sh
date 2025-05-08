#!/bin/bash

# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

# Description: Initializes a Python project with uv and a virtual environment.
# Usage: source ./create_new_project.sh

# Check if sourced
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && { echo -e "\033[0;31m[FAILED] Script must be sourced. Use 'source $0' or '. $0'.\033[0m" >&2; exit 1; }

# Config
EXIT_SUCCESS=0
EXIT_ERROR=1

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
log() {
    case $1 in
        OK) echo -e "${GREEN}[OK]${NC} $2" ;;
        FAILED) echo -e "${RED}[FAILED]${NC} $2" >&2; return $EXIT_ERROR ;;
        INFO) echo -e "${GREEN}[INFO]${NC} $2" ;;
    esac
}

main() {
    log INFO "Starting project creation..."

    # Prompt for project name
    local project_name
    read -p "Enter project name: " project_name
    [[ -z "$project_name" ]] && log FAILED "Project name cannot be empty"

    # Check if directory exists
    [[ -d "$project_name" ]] && log FAILED "Directory '$project_name' already exists. Choose a different name or remove it"

    # Initialize project with uv
    uv init --no-workspace "$project_name" 2>uv_error.log || { log FAILED "Failed to initialize project '$project_name'. See uv_error.log"; return $EXIT_ERROR; }
    rm -f uv_error.log
    log OK "Project '$project_name' initialized"

    # Enter project directory
    cd "$project_name" || log FAILED "Failed to enter directory '$project_name'"
    log OK "Entered project directory"

    # Create virtual environment
    uv venv .venv --prompt "$project_name" || log FAILED "Failed to create virtual environment"
    [[ -f ".venv/bin/activate" ]] || log FAILED "Virtual environment not created properly"
    log OK "Virtual environment created"

    # Activate virtual environment
    source .venv/bin/activate
    log OK "Virtual environment activated"

    # Sync dependencies
    uv sync || log FAILED "Failed to sync dependencies"
    log OK "Dependencies synced"

    log INFO "Project '$project_name' created with virtual environment at '$project_name/.venv'"
}

main
return $EXIT_SUCCESS
