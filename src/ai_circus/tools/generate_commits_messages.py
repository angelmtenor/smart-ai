"""Tool to Generate Commit Messages for AI Circus.
Author: Angel Martinez-Tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import git
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
DEFAULT_LLM_MODEL: str = os.getenv(
    "DEFAULT_LLM_MODEL",
    "gpt-4o-mini" if DEFAULT_LLM_PROVIDER == "openai" else "gemini-2.0-pro",
)
BASE_BRANCH: str = os.getenv("BASE_BRANCH", "main")  # Configurable base branch

async def get_llm() -> Any:
    """Initialize and return the selected LLM based on environment configuration."""
    if DEFAULT_LLM_PROVIDER.lower() == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=DEFAULT_LLM_MODEL, api_key=SecretStr(api_key))
    elif DEFAULT_LLM_PROVIDER.lower() == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, google_api_key=api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {DEFAULT_LLM_PROVIDER}")

def read_styleguide() -> str:
    """Read the styleguide.md file or return a default style guide."""
    styleguide_path = Path("styleguide.md")
    if not styleguide_path.exists():
        return (
            "Use clear, concise commit messages. Start with a verb, describe the "
            "change, and keep it under 72 characters."
        )
    return styleguide_path.read_text()

def get_changed_files() -> list[dict[str, str]]:
    """Retrieve changed files on the current branch compared to the base branch."""
    try:
        repo = git.Repo(".")
        # Check if base branch exists
        if BASE_BRANCH not in repo.heads:
            logger.error(f"Base branch '{BASE_BRANCH}' not found in repository")
            return []

        # Get diff for changed files
        diff = repo.git.diff(f"origin/{BASE_BRANCH}...HEAD", name_status=True)
        changes = []
        for line in diff.splitlines():
            parts = line.split("\t")
            status = parts[0]
            file_path = parts[2] if status.startswith("R") else parts[1]
            # Verify file exists in working tree
            if not Path(file_path).exists():
                logger.warning(f"File {file_path} does not exist in working tree, skipping")
                continue
            # Properly format diff command with '--'
            diff_content = repo.git.diff(f"origin/{BASE_BRANCH}...HEAD", "--", file_path)
            changes.append({"file": file_path, "status": status, "diff": diff_content})
        return changes
    except git.GitCommandError as e:
        logger.error(f"Git error: {e}")
        return []

async def generate_commit_messages(changes: list[dict[str, str]], styleguide: str) -> list[dict[str, Any]]:
    """Group changes and generate commit messages based on the style guide."""
    if not changes:
        logger.info("No changes to process")
        return []

    llm = await get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are a Git expert. Group the following file changes into logical categories
        (e.g., feature, bugfix, refactor, docs) based on their diffs. For each group,
        generate a commit message following the provided style guide. Output the result
        as a JSON list of objects with 'group', 'files', and 'message'.

        **Style Guide**:
        {styleguide}

        **Changes**:
        {changes}

        **Output Format**:
        [
            {{"group": "category", "files": ["file1", "file2"], "message": "message"}},
            ...
        ]
        """
    )

    chain = prompt | llm | StrOutputParser()
    changes_str = "\n".join([f"File: {c['file']}, Status: {c['status']}\nDiff:\n{c['diff']}" for c in changes])
    try:
        result = await chain.ainvoke({"styleguide": styleguide, "changes": changes_str})
        groups = json.loads(result.strip())
        if not isinstance(groups, list):
            logger.error("LLM output is not a valid JSON list")
            return []
        # Validate group structure
        for group in groups:
            if not all(key in group for key in ["group", "files", "message"]):
                logger.error(f"Invalid group structure: {group}")
                return []
        return groups
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing LLM output as JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Error generating commit messages: {e}")
        return []

def write_commit_script(groups: list[dict[str, Any]]) -> Path:
    """Write a shell script with git add and commit commands for each group."""
    script_path = Path("commit_commands.sh")
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
    logger.info("Generated commit commands in %s", script_path)
    with script_path.open("r") as f:
        logger.info("Commands:\n%s", f.read())

    while True:
        choice = input("\nExecute these commands? (yes/no/edit): ").lower()
        if choice == "yes":
            try:
                process = subprocess.Popen(["/bin/bash", str(script_path)])
                process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                logger.info("Commands executed successfully")
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"Error executing commands: {e}")
                break
        elif choice == "no":
            logger.info("Commands not executed. Run them manually from %s", script_path)
            break
        elif choice == "edit":
            logger.info("Please edit %s and run it manually", script_path)
            break
        else:
            logger.warning("Invalid choice. Please enter 'yes', 'no', or 'edit'")

async def main() -> None:
    """Main function to orchestrate commit message generation and execution."""
    try:
        styleguide = read_styleguide()
        changes = get_changed_files()
        if not changes:
            logger.info("No changes found on the current branch")
            return

        groups = await generate_commit_messages(changes, styleguide)
        if not groups:
            logger.info("No commit messages generated")
            return

        script_path = write_commit_script(groups)
        execute_commands(script_path)
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
