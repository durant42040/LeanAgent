"""
Git utilities for LeanAgent.

This module contains functions for cloning repositories, managing branches,
creating pull requests, and finding compatible commits for Lean repositories.
"""

import json
import os
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

import requests
from lean_dojo import LeanGitRepo
from loguru import logger

import generate_benchmark_lean4


def clone_repo(repo_url: str, repo_dir: str) -> Tuple[str, str]:
    """Clone a git repository and return the path to the repository and its sha.

    Args:
        repo_url: The URL of the repository to clone
        repo_dir: The base directory where repositories are stored

    Returns:
        Tuple of (repo_path, sha) where repo_path is the local path to the cloned repo
        and sha is the latest commit hash
    """
    repo_name = "/".join(repo_url.split("/")[-2:]).replace(".git", "")
    logger.info(f"Cloning {repo_url}")
    logger.info(f"Repo name: {repo_name}")
    repo_name = repo_dir + "/" + repo_name
    if os.path.exists(repo_name):
        print(f"Deleting existing repository directory: {repo_name}")
        shutil.rmtree(repo_name)
    subprocess.run(["git", "clone", repo_url, repo_name])
    process = subprocess.Popen(["git", "ls-remote", repo_url], stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    sha = re.split(r"\t+", stdout.decode("utf-8"))[0]
    return repo_name, sha


def branch_exists(repo_name: str, branch_name: str) -> bool:
    """Check if a branch exists in a git repository.

    Args:
        repo_name: Path to the local repository
        branch_name: Name of the branch to check

    Returns:
        True if the branch exists, False otherwise
    """
    proc = subprocess.run(
        ["git", "-C", repo_name, "branch", "-a"], capture_output=True, text=True
    )
    branches = proc.stdout.split("\n")
    local_branch = branch_name
    remote_branch = f"remote/{branch_name}"
    return any(
        branch.strip().endswith(local_branch) or branch.strip().endswith(remote_branch)
        for branch in branches
    )


def create_or_switch_branch(repo_name: str, branch_name: str, base_branch: str) -> None:
    """Create a branch in a git repository if it doesn't exist, or switch to it if it does.

    Args:
        repo_name: Path to the local repository
        branch_name: Name of the branch to create or switch to
        base_branch: Name of the base branch to merge from
    """
    if not branch_exists(repo_name, branch_name):
        subprocess.run(
            ["git", "-C", repo_name, "checkout", "-b", branch_name], check=True
        )
    else:
        subprocess.run(["git", "-C", repo_name, "checkout", branch_name], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                repo_name,
                "merge",
                base_branch,
                "-m",
                f"Merging {branch_name} into {base_branch}",
            ],
            check=True,
        )


def commit_changes(repo_name: str, commit_message: str) -> bool:
    """Commit changes to a git repository.

    Args:
        repo_name: Path to the local repository
        commit_message: The commit message

    Returns:
        True if changes were committed, False if no changes to commit
    """
    status = subprocess.run(
        ["git", "-C", repo_name, "status", "--porcelain"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status == "":
        print("No changes to commit.")
        return False
    subprocess.run(["git", "-C", repo_name, "add", "."], check=True)
    subprocess.run(["git", "-C", repo_name, "commit", "-m", commit_message], check=True)
    return True


def push_changes(repo_name: str, branch_name: str) -> None:
    """Push changes to a git repository.

    Args:
        repo_name: Path to the local repository
        branch_name: Name of the branch to push
    """
    subprocess.run(
        ["git", "-C", repo_name, "push", "-u", "origin", branch_name], check=True
    )


def get_default_branch(repo_full_name: str, personal_access_token: str) -> str:
    """Get the default branch of a repository (default `main`).

    Args:
        repo_full_name: Full name of the repository (e.g., "owner/repo")
        personal_access_token: GitHub personal access token

    Returns:
        The default branch name, or "main" if unable to determine
    """
    url = f"https://api.github.com/repos/{repo_full_name}"
    headers = {
        "Authorization": f"token {personal_access_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["default_branch"]
    else:
        logger.info(f"Failed to get default branch for {repo_full_name}")
        return "main"


def create_pull_request(
    repo_full_name: str,
    title: str,
    body: str,
    head_branch: str,
    personal_access_token: str,
) -> str:
    """Create a pull request in a repository.

    Args:
        repo_full_name: Full name of the repository (e.g., "owner/repo")
        title: Title of the pull request
        body: Body of the pull request
        head_branch: Name of the branch to merge
        personal_access_token: GitHub personal access token

    Returns:
        URL of the created pull request, or empty string if creation failed
    """
    base_branch = get_default_branch(repo_full_name, personal_access_token)
    url = f"https://api.github.com/repos/{repo_full_name}/pulls"
    headers = {
        "Authorization": f"token {personal_access_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"title": title, "body": body, "head": head_branch, "base": base_branch}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        print("Pull request created successfully: " + response.json()["html_url"])
        return response.json()["html_url"]
    else:
        print("Failed to create pull request", response.text)
        return ""


def get_compatible_commit(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Find the most recent commit with a Lean version that LeanAgent supports.

    Args:
        url: The repository URL

    Returns:
        Tuple of (commit_sha, lean_version) or (None, None) if no compatible commit found
    """
    try:
        process = subprocess.Popen(["git", "ls-remote", url], stdout=subprocess.PIPE)
        stdout, stderr = process.communicate()
        latest_commit = re.split(r"\t+", stdout.decode("utf-8"))[0]
        logger.info(f"Latest commit: {latest_commit}")

        new_url = url.replace(".git", "")
        logger.info(f"Creating LeanGitRepo for {new_url}")
        repo = LeanGitRepo(new_url, latest_commit)
        logger.info(f"Getting config for {url}")
        config = repo.get_config("lean-toolchain")
        v = generate_benchmark_lean4.get_lean4_version_from_config(config["content"])
        if generate_benchmark_lean4.is_supported_version(v):
            logger.info(f"Latest commit compatible for url {url}")
            return latest_commit, v

        logger.info(f"Searching for compatible commit for {url}")
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Already in a Git repository")
        except subprocess.CalledProcessError:
            logger.info("Not in a Git repository. Initializing one.")
            subprocess.run(["git", "init"], check=True)

        process = subprocess.Popen(
            ["git", "fetch", "--depth=1000000", url],  # Fetch commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Fetching commits for {url}")
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git fetch command failed: {stderr.decode('utf-8')}")
        logger.info(f"Fetched commits for {url}")
        process = subprocess.Popen(
            ["git", "log", "--format=%H", "FETCH_HEAD"],  # Get list of commits
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"Getting list of commits for {url}")
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise Exception(f"Git log command failed: {stderr.decode('utf-8')}")
        commits = stdout.decode("utf-8").strip().split("\n")
        logger.info(f"Found {len(commits)} commits for {url}")
        for commit in commits:
            new_url = url.replace(".git", "")
            repo = LeanGitRepo(new_url, commit)
            config = repo.get_config("lean-toolchain")
            v = generate_benchmark_lean4.get_lean4_version_from_config(
                config["content"]
            )
            if generate_benchmark_lean4.is_supported_version(v):
                logger.info(f"Found compatible commit {commit} for {url}")
                return commit, v

        raise Exception("No compatible commit found")

    except Exception as e:
        logger.info(f"Error in get_compatible_commit: {str(e)}")
        return None, None


def find_and_save_compatible_commits(
    repo_info_file: str, lean_git_repos: List[LeanGitRepo]
) -> List[dict]:
    """Finds compatible commits for various repositories.

    Args:
        repo_info_file: Path to save the repository information
        lean_git_repos: List of LeanGitRepo objects

    Returns:
        List of dictionaries containing repository information with compatible commits
    """
    updated_repos = []
    for repo in lean_git_repos:
        url = repo.url
        if not url.endswith(".git"):
            url = url + ".git"

        sha = None
        v = None
        if "mathlib4" in url:
            sha = "2b29e73438e240a427bcecc7c0fe19306beb1310"
            v = "v4.8.0"
        elif "SciLean" in url:
            sha = "22d53b2f4e3db2a172e71da6eb9c916e62655744"
            v = "v4.7.0"
        elif "pfr" in url:
            sha = "fa398a5b853c7e94e3294c45e50c6aee013a2687"
            v = "v4.8.0-rc1"
        else:
            sha, v = get_compatible_commit(url)
        if not sha:
            logger.info(f"Failed to find a compatible commit for {url}")
            continue

        updated_repos.append(
            {"url": url.replace(".git", ""), "commit": sha, "version": v}
        )

    with open(repo_info_file, "w") as f:
        json.dump(updated_repos, f)

    return updated_repos


def search_github_repositories(
    language: str = "Lean",
    num_repos: int = 10,
    personal_access_token: str = None,
    known_repositories: List[str] = None,
    repo_dir: str = None,
) -> List[LeanGitRepo]:
    """Search for the given number of repositories on GitHub that have the given language.

    Args:
        language: Programming language to search for
        num_repos: Number of repositories to find
        personal_access_token: GitHub personal access token
        known_repositories: List of repository names to skip
        repo_dir: Directory to clone repositories into

    Returns:
        List of LeanGitRepo objects for the found repositories
    """
    if personal_access_token is None:
        personal_access_token = os.environ.get("GITHUB_ACCESS_TOKEN")

    if known_repositories is None:
        known_repositories = []

    if repo_dir is None:
        repo_dir = os.environ.get("RAID_DIR", "") + "/repos_new"

    headers = {"Authorization": personal_access_token}
    query_params = {
        "q": f"language:{language}",
        "sort": "stars",
        "order": "desc",
        "per_page": 100,
    }

    cloned_count = 0
    page = 1
    lean_git_repos = []

    while cloned_count < num_repos:
        query_params["page"] = page
        response = requests.get(
            "https://api.github.com/search/repositories",
            headers=headers,
            params=query_params,
        )

        if response.status_code == 200:
            repositories = response.json()["items"]
            for repo in repositories:
                if cloned_count >= num_repos:
                    break
                repo_full_name = repo["full_name"]
                logger.info(f"Processing {repo_full_name}")
                if repo_full_name not in known_repositories:
                    name = None
                    try:
                        clone_url = repo["clone_url"]
                        repo_name, sha = clone_repo(clone_url, repo_dir)
                        name = repo_name
                        url = clone_url.replace(".git", "")
                        lean_git_repo = LeanGitRepo(url, sha)
                        lean_git_repos.append(lean_git_repo)
                        cloned_count += 1
                        logger.info(f"Cloned {repo_full_name}")
                    except Exception as e:
                        if name:
                            shutil.rmtree(name)
                        logger.info(f"Failed to clone {repo_full_name} because of {e}")
                else:
                    logger.info(
                        f"Skipping {repo_full_name} since it is a known repository"
                    )
            page += 1
        else:
            logger.info("Failed to search GitHub", response.status_code)
            break

        # Check if we've reached the end of the search results
        if len(repositories) < 100:
            break

    logger.info(f"Total repositories processed: {cloned_count}")
    return lean_git_repos
