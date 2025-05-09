# Dev Container: Data Science Ubuntu 24.04
# Author: Angel Martinez-Tenor (updated by Grok)
# Date: 2025
# Source: https://github.com/angelmtenor/ds-template
# Description: Builds a development container with Git, UV, Cookiecutter, and essential data science tools.
#              Optional GPU (NVIDIA/CUDA) support via build argument.
# Build Arguments:
#   - ENABLE_GPU: Set to "true" to enable NVIDIA/CUDA support (default: false)
#   - USERNAME: Non-root username (default: user)
#   - USER_UID: User ID (default: 1000)
#   - USER_GID: Group ID (default: 1000)

# Base image
FROM ubuntu:24.04

# Metadata
LABEL maintainer="Angel Martinez-Tenor <angelmtenor@gmail.com>" \
    version="2.0" \
    description="Data Science Dev Container with Git, UV, Cookiecutter, and optional GPU support"

# Build arguments
ARG ENABLE_GPU=false
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    PATH=/home/$USERNAME/.local/bin:$PATH

# Install tini as init system for better signal handling
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends tini && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Remove default ubuntu user and install system packages
RUN userdel -r ubuntu 2>/dev/null || true && \
    apt-get update -y && \
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
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Optional GPU support (NVIDIA/CUDA)
RUN if [ "$ENABLE_GPU" = "true" ]; then \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
    nvtop \
    ubuntu-drivers-common \
    nvidia-cuda-toolkit && \
    ubuntu-drivers autoinstall && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# Create non-root user with minimal sudo privileges
RUN groupadd --gid "$USER_GID" "$USERNAME" && \
    useradd --uid "$USER_UID" --gid "$USER_GID" -m "$USERNAME" -s /bin/bash && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:/usr/bin/apt-get,/usr/bin/pipx" > /etc/sudoers.d/"$USERNAME" && \
    chmod 0440 /etc/sudoers.d/"$USERNAME"

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Install user-level tools (UV, Cookiecutter)
RUN pipx install uv && \
    pipx install cookiecutter && \
    pipx ensurepath && \
    rm -rf /home/$USERNAME/.cache/pipx

# Configure Git
RUN git config --global init.defaultBranch main && \
    git config --global core.editor nano

# Create setup script for user – use relative path for alias
RUN echo "source ~/.bashrc" >> /home/$USERNAME/.bash_profile && \
    echo "alias setup='source ./setup.sh'" >> /home/$USERNAME/.bashrc

# Set entrypoint to use tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]

# Suggest using the setup script with relative path
RUN echo "To set up your development environment, run the following command from your home directory:" && \
    echo "  source ./setup_user.sh" && \
    echo "You can also use the alias 'setup' (ensure you're in the correct directory)." && \
    echo "For more information, check the README.md file in the repository."
