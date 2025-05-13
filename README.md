# ai-circus

A Building Block for Generative AI Tools Applications with state-of-the-art performance.

---
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

## Package Information

![PyPI Package](https://img.shields.io/badge/Package%20Version-0.0.1-green?style=for-the-badge)
![Supported Python Versions](https://img.shields.io/badge/Supported%20Python%20Versions-3.13%2B-blue?style=for-the-badge)

---

## Tools and Frameworks


![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=FFD43B)
![uv](https://img.shields.io/badge/uv-4baaaa?style=for-the-badge&logo=github)
![Ruff](https://img.shields.io/badge/Ruff-000000?style=for-the-badge&logo=ruff&logoColor=white)
![Pyright](https://img.shields.io/badge/Pyright-61DAFB?style=for-the-badge&logo=pyright&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9DFF?style=for-the-badge&logo=pytest&logoColor=white)
![Pre-commit](https://img.shields.io/badge/Pre--commit-FDA50F?style=for-the-badge&logo=pre-commit&logoColor=white)

---

## Getting Started – Package Usage

This project requires Python 3.13 or later. We highly recommend using a virtual environment to manage dependencies and avoid conflicts with system packages. Follow these steps to get started:

1.  Install the package using pip:
    ```bash
    pip install ai-circus
    ```
    Alternatively, if you prefer using uv, run:
    ```bash
    uv add ai-circus
    ```

2.  Set up your environment configuration:
    - Copy the provided template file to create your environment settings:
      ```bash
      cp env.example .env
      ```
    - Open the newly created `.env` file and update the required environment variables (the OpenAI key is mandatory).
    3. Test the available tools:

    Open your terminal and run the following commands associated with the import:

    - `ai-hello-world`: Executes a simple "Hello, World!" example.
    - `ai-check-api-key`: Verifies the settings in your `.env` file by checking the OpenAI, Google, and Tavily API keys.
    - `ai-commit`: Automates the process of committing changes in your current repository.

## Development Setup

To set up your machine for development (recommended for Debian/Ubuntu-based systems), execute the following commands in the root directory of your cloned project:

- Prepare your system by running the appropriate setup scripts.
- These scripts will install necessary dependencies and configure your development environment.

This setup ensures that your machine is properly configured and ready for efficient development.


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

After the one-time setup, use the `setup` macro to ensure the Python environment for the project is correctly recreated and synced (it also executes the below `make qa` command).

```bash
setup
```

This command performs the following actions:

*   Checks and activates the virtual environment.
*   Syncs project dependencies using `uv`.
*   Installs pre-commit hooks if not already installed and runs them (`make qa`)

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

**Contributing Guidelines:**
- Review our [Style Guide](styleguide.md) to ensure clear and descriptive commit messages and consistent coding standards.
- See our [Contribution Guidelines](CONTRIBUTING.md) for details on the workflow and submission process.
- Please abide by our [Code of Conduct](CODE_OF_CONDUCT.md) to help maintain a welcoming and collaborative community.
