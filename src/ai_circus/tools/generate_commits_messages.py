"""Tool to generate commit messages for staged, unstaged, and untracked files in a Git repository.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
from ai_circus.core import custom_logger
from ai_circus.models import get_llm

logger = custom_logger.init(level="INFO")
# Load environment variables

# # Configuration
# DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
# DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
# BASE_BRANCH = os.getenv("BASE_BRANCH", "main")


# async def get_llm() -> Any:
#     """Initialize and return the selected LLM."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY not set")
#     return ChatOpenAI(model=DEFAULT_LLM_MODEL, api_key=SecretStr(api_key))


def read_styleguide() -> str:
    """Read the styleguide.md file or return a default style guide."""
    styleguide_path = Path("styleguide.md")
    default_styleguide = (
        "Use clear, concise commit messages. Start with a verb, describe the change, and keep it under 72 characters."
    )
    return styleguide_path.read_text() if styleguide_path.exists() else default_styleguide


def run_git_command(command: list[str]) -> str:
    """Run a Git command and return its output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)  # noqa: S603
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        return ""


def get_changed_files() -> list[dict[str, str]]:
    """Retrieve uncommitted changes (staged, unstaged, and untracked files)."""
    changes = []

    # Staged changes
    staged_files = run_git_command(["git", "diff", "--cached", "--name-status"])
    for line in staged_files.splitlines():
        status, file_path = line.split("\t", 1)
        if Path(file_path).exists():
            diff_content = run_git_command(["git", "diff", "--cached", "--", file_path])
            if diff_content:
                diff_lines = diff_content.splitlines()[:10]
                diff_content = "\n".join(diff_lines) + "\n... (truncated)"
            changes.append({"file": file_path, "status": status, "diff": diff_content})

    # Unstaged changes
    unstaged_files = run_git_command(["git", "diff", "--name-status"])
    for line in unstaged_files.splitlines():
        status, file_path = line.split("\t", 1)
        if Path(file_path).exists():
            diff_content = run_git_command(["git", "diff", "--", file_path])
            if diff_content:
                diff_lines = diff_content.splitlines()[:10]
                diff_content = "\n".join(diff_lines) + "\n... (truncated)"
            changes.append({"file": file_path, "status": status, "diff": diff_content})

    # Untracked files
    untracked_files = run_git_command(["git", "ls-files", "--others", "--exclude-standard"]).splitlines()
    for file_path in untracked_files:
        if Path(file_path).exists():
            changes.append({"file": file_path, "status": "A", "diff": "New file added"})

    # Remove duplicates
    unique_changes = list({change["file"]: change for change in changes}.values())
    logger.debug(f"Detected changes: {unique_changes}")
    return unique_changes


async def generate_commit_messages(changes: list[dict[str, str]], styleguide: str) -> list[dict[str, Any]]:
    """Generate commit messages for the given changes."""
    if not changes:
        logger.info("No changes to process")
        return []

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Git expert. Based on the following file change, generate a single commit message
        summarizing all changes in the file, following the provided style guide. Classify the change
        into a logical category (e.g., feature, bugfix, refactor, docs, setup).

        **Style Guide**:
        {styleguide}

        **Change**:
        File: {file}
        Status: {status}
        Diff (may be truncated to first 10 lines):
        {diff}

        **Output Format**:
        Your response must be only a JSON object in the following format:
        {{"group": "category", "files": ["{file}"], "message": "commit message summarizing all changes"}}
        Do not include any additional text, code blocks, explanations, or comments.
        Ensure that the output is a valid JSON object.

        **Important**: Do not generate a commit message in plain text. Only return the JSON object as specified.
        """
    )
    chain = prompt | llm | StrOutputParser()

    async def process_change(change: dict[str, str]) -> dict[str, Any] | None:
        """
        Process a single file change and generate a commit message.

        Args:
            change (dict): A dictionary containing file change information.

        Returns:
            dict: A dictionary containing the generated commit message and group.
        """
        result = await chain.ainvoke(
            {"styleguide": styleguide, "file": change["file"], "status": change["status"], "diff": change["diff"]}
        )
        logger.info(f"Raw LLM output for {change['file']}: {result}")
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if not json_match:
            logger.error(f"No JSON found in LLM output for {change['file']}")
            return None
        try:
            group = json.loads(json_match.group(0))
            if not all(key in group for key in ["group", "files", "message"]):
                logger.error(f"Invalid JSON structure for {change['file']}: {group}")
                return None
            if group["files"] != [change["file"]]:
                logger.error(f"Incorrect files in JSON for {change['file']}: {group['files']}")
                return None
            return group
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {change['file']}: {e}")
            return None

    tasks = [process_change(change) for change in changes]
    groups = await asyncio.gather(*tasks)
    return [group for group in groups if group is not None]


def write_commit_script(groups: list[dict[str, Any]]) -> Path:
    """Write a shell script with git commands.

    Args:
        groups (list): A list of dictionaries containing commit messages and file groups.

    Returns:
        Path: The path to the generated shell script.


    """
    script_path = Path("temp_output/commit_commands.sh")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with script_path.open("w") as f:
        f.write("#!/bin/bash\n\n")
        for group in groups:
            f.write(f"# {group['group']}\n")
            for file in group["files"]:
                f.write(f"git add {file}\n")
            f.write(f'git commit -m "{group["message"]}"\n\n')
    script_path.chmod(0o755)
    return script_path


def execute_commands(script_path: Path) -> None:
    """Execute the generated commit commands interactively."""
    logger.info(f"Generated commit commands in {script_path}")
    with script_path.open("r") as f:
        logger.info(f"Commands:\n{f.read()}")

    while True:
        choice = input("\nExecute these commands? (yes/no/edit): ").lower()
        if choice == "yes":
            try:
                # Apply a git reset before any git add commands
                subprocess.run(["git", "reset"], check=True)  # noqa: S603, S607
                logger.info("git reset executed successfully")
                subprocess.run(["/bin/bash", str(script_path)], check=True)  # noqa: S603
                logger.info("Commands executed successfully")
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"Error executing commands: {e}")
                break
        elif choice == "no":
            logger.info(f"Commands not executed. Run them manually from {script_path}")
            break
        elif choice == "edit":
            logger.info(f"Please edit {script_path} and run it manually")
            break
        else:
            logger.warning("Invalid choice. Please enter 'yes', 'no', or 'edit'")


async def main() -> None:
    """Main function to generate and execute commit messages."""
    try:
        styleguide = read_styleguide()
        changes = get_changed_files()
        if not changes:
            logger.info("No changes found")
            return

        groups = await generate_commit_messages(changes, styleguide)
        if not groups:
            logger.info("No commit messages generated")
            return

        script_path = write_commit_script(groups)
        execute_commands(script_path)
    except Exception as e:
        logger.error(f"Error: {e}")


def run_main() -> None:
    """Run the main function with asyncio (to be used as tool)"""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
