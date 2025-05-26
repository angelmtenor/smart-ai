#!/bin/bash

# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
# Description: Sets up or updates user environment with Git, tools, and shell config.
# Usage: source setup_user.sh [--update]

set -euo pipefail
trap 'echo -e "\e[31m❌ Failed at line $LINENO\e[0m" >&2; exit 1' ERR

# Configuration
BASHRC="$HOME/.bashrc"
LOCAL_BIN="$HOME/.local/bin"
NVM_DIR="$HOME/.nvm"

# Logging functions
log() { echo -e "\e[32m✅ $1\e[0m"; }
err() { echo -e "\e[31m❌ $1\e[0m" >&2; exit 1; }
info() { echo -e "\e[34mℹ $1\e[0m"; }

# Parse arguments
UPDATE_MODE=false
[[ "${1:-}" == "--update" ]] && UPDATE_MODE=true

# Append content to .bashrc if not already present
append_to_bashrc() {
    local content="$1" comment="$2"
    if ! grep -qF "$comment" "$BASHRC"; then
        echo -e "$comment\n$content" >> "$BASHRC"
        log "Added to $BASHRC: $comment"
    elif [[ "$UPDATE_MODE" == true ]]; then
        read -p "Overwrite $comment? (y/N): " overwrite
        if [[ "$overwrite" =~ ^[Yy]$ ]]; then
            sed -i "/$comment/,+1d" "$BASHRC"
            echo -e "$comment\n$content" >> "$BASHRC"
            log "Updated $comment in $BASHRC."
        else
            info "$comment already in $BASHRC."
        fi
    else
        info "$comment already in $BASHRC."
    fi
}

# Ensure ~/.local/bin is in PATH
ensure_path() {
    if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
        export PATH="$LOCAL_BIN:$PATH"
        log "Added $LOCAL_BIN to PATH."
    fi
    append_to_bashrc 'export PATH="$HOME/.local/bin:$PATH"' "# PATH from setup_user.sh"
}

# Configure Git
configure_git() {
    command -v git &>/dev/null || err "Git not found. Please install Git."
    git config --global init.defaultBranch main
    local name=$(git config --global user.name)
    local email=$(git config --global user.email)
    if [[ "$UPDATE_MODE" == true || -z "$name" || -z "$email" ]]; then
        info "Current Git config - name: $name, email: $email."
        read -p "Enter Git username (press Enter to keep): " new_name
        read -p "Enter Git email (press Enter to keep): " new_email
        [[ -n "$new_name" ]] && git config --global user.name "$new_name" && log "Set Git username to: $new_name."
        [[ -n "$new_email" ]] && git config --global user.email "$new_email" && log "Set Git email to: $new_email."
    else
        log "Git already configured with name: $name, email: $email."
    fi
}

# Set custom prompt
set_custom_prompt() {
    local prompt='export PS1="\[\e[32m\]\u@\h \[\e[34m\]\w\[\e[0m\]\$ "'
    append_to_bashrc "$prompt" "# Custom prompt from setup_user.sh"
    eval "$prompt"
    log "Custom prompt set for current shell."
}

# Add setup alias
add_alias() {
    local alias_cmd="alias setup='source setup_user.sh'"
    append_to_bashrc "$alias_cmd" "# Alias from setup_user.sh"
    eval "$alias_cmd"
    log "Alias 'setup' set for current shell."
}

# Install or update nvm and Node.js 20
install_nvm() {
    local latest_nvm_version=$(curl -s https://api.github.com/repos/nvm-sh/nvm/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
    if [[ -d "$NVM_DIR" && -s "$NVM_DIR/nvm.sh" ]]; then
        if [[ "$UPDATE_MODE" == true ]]; then
            local current_nvm_version=$(grep 'nvm_version=' "$NVM_DIR/nvm.sh" | cut -d'=' -f2 | tr -d '"')
            if [[ "$current_nvm_version" != "$latest_nvm_version" ]]; then
                log "Updating nvm to $latest_nvm_version..."
                curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/$latest_nvm_version/install.sh" | bash || err "Failed to update nvm."
                log "nvm updated to $latest_nvm_version."
            else
                info "nvm is up-to-date ($current_nvm_version)."
            fi
        else
            log "nvm is already installed."
        fi
    else
        log "Installing nvm $latest_nvm_version..."
        curl -o- "https://raw.githubusercontent.com/nvm-sh/nvm/$latest_nvm_version/install.sh" | bash || err "Failed to install nvm."
        log "nvm installed."
    fi
    . "$NVM_DIR/nvm.sh"

    if nvm ls 20 &>/dev/null; then
        [[ "$UPDATE_MODE" == true ]] && nvm install 20 --reinstall-packages-from=20 && log "Node.js 20 updated." || log "Node.js 20 is already installed."
    else
        log "Installing Node.js 20..."
        nvm install 20 || err "Failed to install Node.js 20."
        log "Node.js 20 installed."
    fi

    [[ $(nvm alias default) != *"20"* ]] && nvm alias default 20 && log "Set Node.js 20 as default." || log "Node.js 20 is already default."
}

# Install or update uv
install_uv() {
    if command -v uv &>/dev/null; then
        if [[ "$UPDATE_MODE" == true ]]; then
            local current_uv_version=$(uv --version | cut -d' ' -f2)
            local latest_uv_version=$(curl -s https://api.github.com/repos/astral-sh/uv/releases/latest | grep '"tag_name"' | cut -d'"' -f4 | sed 's/^v//')
            if [[ "$current_uv_version" != "$latest_uv_version" ]]; then
                log "Updating uv to $latest_uv_version..."
                curl -LsSf https://astral.sh/uv/install.sh | sh || err "Failed to update uv."
                log "uv updated to $latest_uv_version."
            else
                info "uv is up-to-date ($current_uv_version)."
            fi
        else
            log "uv is already installed."
        fi
    else
        log "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || err "Failed to install uv."
        log "uv installed."
    fi
}

# Main function
main() {
    command -v curl &>/dev/null || err "curl not found. Please install curl."
    log "Starting $([[ "$UPDATE_MODE" == true ]] && echo "update" || echo "setup")..."
    ensure_path
    configure_git
    set_custom_prompt
    add_alias
    install_nvm
    install_uv
    log "$([[ "$UPDATE_MODE" == true ]] && echo "Update" || echo "Setup") completed. Run 'source ~/.bashrc' to apply changes in future shells."
}

main "$@"
