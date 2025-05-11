#!/bin/bash

# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

# Description: Sets up or updates user environment with Git, tools, and shell config for dual environments.
# Usage: source setup_user.sh [--update]

set -euo pipefail
trap 'echo -e "\e[31m❌ Failed at line $LINENO\e[0m" >&2; exit 1' ERR

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
        error) echo -e "${RED}❌ $2${NC}" >&2; exit 1 ;;
        info) echo -e "${GREEN}ℹ️ $2${NC}" ;;
    esac
}

check_cmd() { command -v "$1" &>/dev/null; }
ensure_dir() { mkdir -p "$1"; }

# Check internet
check_connectivity() {
    log info "Checking internet..."
    ping -c 1 -W 2 8.8.8.8 &>/dev/null || log error "No internet."
    log success "Internet verified."
}

# Install and configure pipx
ensure_pipx() {
    check_cmd pipx && { log success "pipx already configured."; return; }
    log info "Installing pipx..."
    python3 -m pip install --user pipx &>/dev/null || log error "Failed to install pipx."
    pipx ensurepath &>/dev/null
    source ~/.bashrc &>/dev/null || true
    log success "pipx configured."
}

# Manage Python tools
manage_tools() {
    local mode="$1" tools=(uv cookiecutter pre-commit)
    ensure_pipx
    for tool in "${tools[@]}"; do
        if check_cmd "$tool" && [[ "$mode" == "update" ]]; then
            log info "Upgrading $tool..."
            pipx upgrade "$tool" &>/dev/null || log error "Failed to upgrade $tool."
            log success "$tool upgraded."
        elif check_cmd "$tool"; then
            log success "$tool already installed."
        else
            log info "Installing $tool..."
            pipx install "$tool" &>/dev/null || log error "Failed to install $tool."
            log success "$tool installed."
        fi
    done
}

# Configure Git
configure_git() {
    check_cmd git || log error "Git not found. Install with 'sudo apt install git'."
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    local name email
    name=$(git config --global user.name || true)
    email=$(git config --global user.email || true)
    if [[ -z "$name" || -z "$email" ]]; then
        [[ -t 0 ]] || log error "Git config missing. Set manually:\n  git config --global user.name \"Your Name\"\n  git config --global user.email \"you@example.com\""
        while [[ -z "$name" ]]; do
            read -p "Git username: " name
            [[ -z "$name" ]] && log warn "Username cannot be empty."
        done
        while [[ -z "$email" || ! "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; do
            read -p "Git email: " email
            [[ -z "$email" ]] && log warn "Email cannot be empty."
            [[ ! "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]] && log warn "Invalid email format."
        done
        git config --global user.name "$name"
        git config --global user.email "$email"
    fi
    log success "Git configured: $name <$email>."
}

# Customize Bash prompt
customize_bash_prompt() {
    local bashrc=~/.bashrc marker="# Custom prompt"
    grep -qF "$marker" "$bashrc" 2>/dev/null && { log success "Prompt already customized."; return; }
    cat <<'EOF' >>"$bashrc"
# Custom prompt
RED='\[\e[31m\]'
GREEN='\[\e[32m\]'
YELLOW='\[\e[33m\]'
BLUE='\[\e[34m\]'
CYAN='\[\e[36m\]'
RESET='\[\e[0m\]'

parse_git_branch() { git branch 2>/dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'; }
prompt_venv() { [[ -n "${VIRTUAL_ENV}" ]] && echo "(${VIRTUAL_ENV##*/}) "; }
export PS1="\$(prompt_venv)${GREEN}\u${BLUE}@\h ${CYAN}\W${YELLOW}\$(parse_git_branch)${RESET}\$ "
EOF
    log success "Prompt customized."
}

# Configure alias
configure_alias() {
    local bashrc=~/.bashrc alias_cmd="alias setup='source setup.sh'"
    grep -qF "$alias_cmd" "$bashrc" 2>/dev/null && { log success "Alias 'setup' already configured."; return; }
    echo "$alias_cmd" >>"$bashrc"
    log success "Alias 'setup' added."
}

# Create projects directory
create_projects_dir() {
    ensure_dir "$HOME/PROJECTS"
    log success "Projects directory at $HOME/PROJECTS."
}

# Verify tools
verify_installations() {
    log info "Verifying tools..."
    for cmd in git uv cookiecutter pre-commit; do
        check_cmd "$cmd" || log error "$cmd not installed."
    done
    log success "Tools verified."
}

# Install nvm and Node.js 20
setup_nvm() {
    log info "Checking nvm and Node.js 20..."
    check_cmd curl || log error "curl not found. Install with 'sudo apt install curl'."

    # Check if nvm is installed
    if [[ -d "$HOME/.nvm" && -s "$HOME/.nvm/nvm.sh" ]]; then
        log success "nvm already installed."
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    else
        log info "Installing nvm..."
        curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        [[ $? -ne 0 ]] && log error "Failed to install nvm."
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        log success "nvm installed."
    fi

    # Check if Node.js 20 is installed
    if nvm ls 20 &>/dev/null; then
        log success "Node.js 20 already installed."
        nvm use 20 &>/dev/null || log error "Failed to activate Node.js 20."
    else
        log info "Installing Node.js 20..."
        nvm install 20 &>/dev/null || log error "Failed to install Node.js 20."
        nvm use 20 &>/dev/null || log error "Failed to activate Node.js 20."
        log success "Node.js 20 installed."
    fi

    log success "nvm and Node.js 20 configured."
}

# Main
main() {
    local mode="install"
    [[ "${1:-}" == "--update" ]] && mode="update"
    [[ $# -gt 1 ]] && log error "Usage: $0 [--update]"
    log info "Starting $mode..."
    check_connectivity
    manage_tools "$mode"
    if [[ "$mode" == "install" ]]; then
        configure_git
        create_projects_dir
        customize_bash_prompt
        configure_alias
        log warn "Run 'source ~/.bashrc' to apply changes."
    fi

    setup_nvm

    # Add the directory where pipx installs binaries to PATH so that 'uv'
    # and other tools are available in the current session
    export PATH="$HOME/.local/bin:$PATH"

    verify_installations
    log success "Environment $mode completed."

}

main "$@"
