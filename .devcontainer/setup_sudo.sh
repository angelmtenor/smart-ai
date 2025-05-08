#!/bin/bash

# Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template

# Updates Ubuntu system and installs essential project packages.
# Optionally installs NVIDIA/CUDA GPU support.
# Recommended: Ubuntu 24.04 LTS or later
# Usage: chmod +x setup_sudo.sh && sudo ./setup_sudo.sh

# Exit codes
EXIT_SUCCESS=0
EXIT_NOT_ROOT=1
EXIT_ERROR=2

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check root
[[ $EUID -ne 0 ]] && { echo -e "${RED}❌ Failure: Run as root${NC}" >&2; exit $EXIT_NOT_ROOT; }

# Error handling
set -e
trap 'echo -e "${RED}❌ Failure: Failed at line $LINENO${NC}" >&2; exit $EXIT_ERROR' ERR

# Logging with standardized indicators
log() {
    case $1 in
        INFO) echo -e "${GREEN}ℹ️ Info: $2${NC}" ;;
        WARN) echo -e "${YELLOW}⚠️ Warning: $2${NC}" ;;
        ERROR) echo -e "${RED}❌ Failure: $2${NC}" >&2 ;;
        DEBUG) echo -e "🔍 Debug: $2" ;;
        SUCCESS) echo -e "${GREEN}✅ Success: $2${NC}" ;;
    esac
}

# Check internet connectivity
check_connectivity() {
    log INFO "Checking internet connectivity..."
    if ! ping -c 1 -W 2 8.8.8.8 &> /dev/null; then
        log ERROR "No internet connection detected."
        exit $EXIT_ERROR
    fi
    log SUCCESS "Internet connection verified."
}

# Add NVIDIA repository
add_nvidia_repository() {
    log INFO "Adding NVIDIA repository..."
    if ! command -v add-apt-repository &> /dev/null; then
        apt install -y software-properties-common
    fi
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt update -y
    log SUCCESS "NVIDIA repository added."
}

# Install base packages
install_base_packages() {
    log INFO "Installing base packages..."
    apt update -y
    apt full-upgrade -y
    apt install -y --no-install-recommends \
        git git-flow make curl wget ca-certificates \
        nano htop gcc g++ clang linux-libc-dev pipx xclip \
        python3 python3-pip python3-venv
    apt autoremove -y
    log SUCCESS "Base packages installed."
}

# Install GPU support
install_gpu_support() {
    log INFO "Installing NVIDIA/CUDA support..."
    if ! command -v nvidia-smi &> /dev/null; then
        log DEBUG "No NVIDIA GPU detected, attempting driver installation..."
        add_nvidia_repository
        apt install -y nvtop ubuntu-drivers-common nvidia-driver-latest nvidia-cuda-toolkit
        ubuntu-drivers autoinstall
    else
        log INFO "NVIDIA GPU detected, installing CUDA toolkit..."
        apt install -y nvtop nvidia-cuda-toolkit
    fi
    log SUCCESS "GPU support installed."
    log WARN "System will reboot in 5 seconds (press Ctrl+C to cancel)..."
    sleep 5
    reboot
}

# Verify installations
verify_installations() {
    log INFO "Verifying installations..."
    for cmd in git curl wget nano htop gcc g++ clang pipx xclip python3; do
        if command -v "$cmd" &> /dev/null; then
            log SUCCESS "$cmd is installed."
        else
            log ERROR "$cmd installation failed."
            exit $EXIT_ERROR
        fi
    done
    log SUCCESS "All base packages verified."
}

# Main
main() {
    log INFO "Starting setup..."
    check_connectivity
    install_base_packages
    verify_installations
    echo -e "${YELLOW}[INPUT REQUIRED] Install NVIDIA/CUDA support? [y/N]: ${NC}"
    read -r -n 1 reply
    echo
    [[ $reply =~ ^[Yy]$ ]] && install_gpu_support || log INFO "Skipping GPU support."
    log SUCCESS "Setup completed."
}

main
exit $EXIT_SUCCESS
