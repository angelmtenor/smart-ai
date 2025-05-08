# ai-circus

A Building Block for Generative AI Applications with state-of-the-art performance.

---


## Package Information

![PyPI Package](https://img.shields.io/badge/Package%20Version-0.0.1-green?style=for-the-badge)
![Supported Python Versions](https://img.shields.io/badge/Supported%20Python%20Versions-3.13%2B-blue?style=for-the-badge)

---

## Tools and Frameworks


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Ruff](https://img.shields.io/badge/Ruff-000000?style=for-the-badge&logo=ruff&logoColor=white)
![Pyright](https://img.shields.io/badge/Pyright-20232A?style=for-the-badge&logo=pyright&logoColor=61DAFB)
![Pytest](https://img.shields.io/badge/Pytest-0A9DFF?style=for-the-badge&logo=pytest&logoColor=white)

---

## Setup

To properly set up your machine for development, follow these steps. These steps are designed for Debian/Ubuntu-based systems. Execute these commands in the root folder of the cloned project.

### One-Time Setup

1.  **Sudo Setup**:

    ```bash
    sudo ./.devcontainer/setup_sudo.sh
    ```

    This script configures `sudo` and installs essential base packages.

2.  **User Setup**:

    ```bash
    ./.devcontainer/setup_user.sh
    ```

    This script configures the user environment, installs `uv`, `cookiecutter`, and `pre-commit`, customizes the prompt, and sets up aliases.

3.  **Update Terminal**:

    ```bash
    source ~/.bashrc
    ```

    Apply the changes made by the user setup script to your current terminal session.

### Project Setup

After the one-time setup, use the `setup` macro to ensure the Python environment for the project is correctly recreated and synced.

```bash
setup
```

This command performs the following actions:

*   Checks and activates the virtual environment.
*   Syncs project dependencies using `uv`.

### Contributing

With the environment set up, you can use tools like `make` to contribute to the project.

*   **Quality Assurance**:

    ```bash
    make qa
    ```

    Runs quality assurance checks, including whitespace trimming, line ending fixes, TOML/YAML/JSON checks, merge conflict checks, Ruff, Pyright, and Bandit.
*   **Build**:

    ```bash
    make build
    ```

    Executes the build process for the project.



## TODO: Script Improvements

- Use `echo "✅ Success"` to indicate success.
- Use `echo "❌ Failure"` to indicate failure.
- Use `echo "⚠️ Warning"` to indicate warning.
- Use `echo "ℹ️ Info"` to indicate info.
- Use `echo "🔍 Debug"` to indicate debug.

---

Feel free to expand this README with additional sections like **Installation**, **Usage**, **Contributing**, and **License** as needed.
