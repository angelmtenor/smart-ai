# Contributing Guidelines

Thank you for considering contributing to this project! We welcome all contributions, whether it's reporting bugs, suggesting enhancements, or submitting code changes. These guidelines aim to ensure a smooth review and integration process while maintaining a welcoming and inclusive environment for all contributors.

Before you start, please take a moment to read our [Code of Conduct](CODE_OF_CONDUCT.md). It outlines our expectations for participation and helps ensure a positive experience for everyone.

## How to Contribute

If you're new to contributing, don't worry! We're here to help. Here are the general steps to follow:

1. **Discuss the Change**: Before making significant changes, it's a good idea to discuss them with the project maintainers. You can do this by opening an issue or starting a discussion.
2. **Fork the Repository**: Create a fork of this repository on GitHub.
3. **Clone Your Fork**: Clone your forked repository to your local machine.
4. **Create a Feature Branch**: Create a new branch for your changes.
5. **Make Changes**: Implement your changes, following the project's coding standards and guidelines.
6. **Run Tests**: Ensure your changes pass all tests. You can run tests using `make test`.
7. **Commit Your Changes**: Commit your changes with clear and descriptive messages.
8. **Push Your Branch**: Push your feature branch to your fork on GitHub.
9. **Open a Pull Request**: Open a pull request from your feature branch to the main branch of the original repository.
10. **Follow Up**: Respond to any feedback or requests for changes from the maintainers.

## Setting Up Your Environment

To set up your local development environment:

1. **Fork the repository** on GitHub.
2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/ai-circus
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b descriptive-feature-name
   ```

## Code Quality and Testing

- Run pre-commit checks:
  ```bash
  make qa
  ```
- Run tests:
  ```bash
  make test
  ```

## Pull Request Process

When opening a pull request, please:

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new environment variables, exposed ports, useful file locations, and container parameters.
- Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
- You may merge the Pull Request once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

## Code of Conduct

By participating in this project, you agree to abide by its [Code of Conduct](CODE_OF_CONDUCT.md).
