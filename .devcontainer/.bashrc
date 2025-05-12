# ~/.bashrc: Bash configuration for interactive, non-login shells
# Author: Angel Martinez-Tenor, 2025
#
# Purpose:
# Configures Bash with history settings, customized prompts, aliases, and environment
# variables for a portable, developer-friendly experience. This file is generated or
# customized by ./.devcontainer/setup_user.sh in a Devcontainer environment.
#
# Usage:
# - Back up your existing ~/.bashrc:
#   cp ~/.bashrc ~/.bashrc.bak
# - Copy this file to your home directory:
#   cp .devcontainer/.bashrc ~/.bashrc
# - Source the file to apply changes:
#   source ~/.bashrc
# - (Optional) Customize ~/.bash_aliases for additional settings.
#
# Prerequisites:
# - Bash 4.0 or later (for features like histappend).
# - Linux or macOS with standard tools (e.g., ls, grep, git).
# - Optional: NVM (Node Version Manager) and pipx for referenced paths.
#
# Notes:
# - Ensure referenced files (e.g., ~/.bash_aliases, setup.sh) exist.
# - Review ~/.bash_aliases and setup.sh for sensitive data before sharing.
# - Portable across machines using $HOME for paths.
# - Verify setup by running `bash --rcfile ~/.bashrc` after sourcing.

# Exit if non-interactive
[[ $- != *i* ]] && return

# History settings
HISTCONTROL=ignoreboth
shopt -s histappend
HISTSIZE=1000
HISTFILESIZE=2000

# Shell options
shopt -s checkwinsize

# Enable lesspipe for non-text files
[[ -x /usr/bin/lesspipe ]] && eval "$(SHELL=/bin/sh lesspipe)"

# Set chroot variable if applicable
if [[ -z "${debian_chroot:-}" && -r /etc/debian_chroot ]]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# Enable color prompt for supported terminals
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes ;;
esac

# Define prompt
if [[ "$color_prompt" = yes ]]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\e[32m\]\u\[\e[34m\]@\h \[\e[36m\]\W\[\e[33m\]$(git branch 2>/dev/null | sed -e "/^[^*]/d" -e "s/* \(.*\)/ (\1)/")\[\e[0m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w$(git branch 2>/dev/null | sed -e "/^[^*]/d" -e "s/* \(.*\)/ (\1)/")\$ '
fi

# Set terminal title for xterm
case "$TERM" in
    xterm*|rxvt*)
        PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
        ;;
esac

# Enable color support for ls and grep
if [[ -x /usr/bin/dircolors ]]; then
    [[ -r ~/.dircolors ]] && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# Common aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Load bash aliases if present
[[ -f ~/.bash_aliases ]] && . ~/.bash_aliases

# Enable bash completion
if ! shopt -oq posix; then
    if [[ -f /usr/share/bash-completion/bash_completion ]]; then
        . /usr/share/bash-completion/bash_completion
    elif [[ -f /etc/bash_completion ]]; then
        . /etc/bash_completion
    fi
fi

# Add pipx to PATH
export PATH="$PATH:$HOME/.local/bin"

# Project setup alias
alias setup='source setup.sh'

# Custom prompt colors
RED='\[\e[31m\]'
GREEN='\[\e[32m\]'
YELLOW='\[\e[33m\]'
BLUE='\[\e[34m\]'
CYAN='\[\e[36m\]'
RESET='\[\e[0m\]'

# NVM setup
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
