# Dev Container: Data Science Ubuntu 24.04
# Author: Angel Martinez-Tenor
# Date: 2022-2024
# Source: https://github.com/angelmtenor/ds-template
# Description: Builds a development container with Git, UV, Cookiecutter, and essential tools.
#              Supports GPU (NVIDIA/CUDA) with commented-out instructions.
# Notes:
# - Based on Ubuntu 24.04; adjust for other versions if needed.
# - Installs tools as a non-root user for compatibility with VS Code dev containers.
# - GPU support requires NVIDIA Container Toolkit and uncommented GPU section.

# Base image
FROM ubuntu:24.04

# Set environment variables for consistent locale
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Remove default ubuntu user (specific to Ubuntu 24.04)
RUN touch /var/mail/ubuntu && chown ubuntu /var/mail/ubuntu && userdel -r ubuntu

# Install system packages and tools (root user)
RUN apt-get update --fix-missing -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        git \
        git-flow \
        make \
        curl \
        wget \
        ca-certificates \
        nano \
        htop \
        gcc \
        g++ \
        clang \
        linux-libc-dev \
        pipx \
        unzip \
        zip \
        xclip \
        ffmpeg \
        libsm6 \
        libxext6 \
        sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# GPU Support (NVIDIA/CUDA) - Uncomment to enable
# Requires NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
# RUN apt-get update -y && \
#     apt-get install -y --no-install-recommends \
#         nvtop \
#         ubuntu-drivers-common \
#         nvidia-cuda-toolkit && \
#     ubuntu-drivers autoinstall && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Create non-root user
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid "$USER_GID" "$USERNAME" && \
    useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/"$USERNAME" && \
    chmod 0440 /etc/sudoers.d/"$USERNAME"

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Configure shell and environment
ENV SHELL=/usr/bin/bash \
    PATH=/home/$USERNAME/.local/bin:$PATH

# Install user-level tools (UV, Cookiecutter)
RUN pipx install uv && \
    pipx install cookiecutter && \
    pipx ensurepath

# Configure Git
RUN git config --global init.defaultBranch main

# Set default command
CMD ["/usr/bin/bash"]
