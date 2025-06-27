"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

import argparse
import hashlib
import json
import os
import pickle
import uuid
from typing import List, Optional, Tuple

from lean_dojo import LeanGitRepo, Pos, Theorem, is_available_in_cache
from loguru import logger

from common import set_logger
from prover.proof_search import DistributedProver, Status


def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    """
    Retrieves a list of Lean theorems from specified files based on given filters.

    This function fetches theorems from Lean files using internal helper functions and
    validates that all repositories containing the theorems have been traced with LeanDojo.

    Parameters:
        data_path (str): Path to the data directory containing Lean files.
        split (str): Dataset split identifier (e.g., 'train', 'valid', 'test').
        file_path (str): Path to specific file or directory of files to search.
        full_name (str): Full name pattern to filter theorems.
        name_filter (str): Name pattern to filter theorems.
        num_theorems (int): Maximum number of theorems to retrieve.

    Returns:
        Tuple[LeanGitRepo, List[Theorem], List[Pos]]: A tuple containing:
            - LeanGitRepo: The repository object
            - List[Theorem]: List of theorem objects
            - List[Pos]: List of theorem positions

    Raises:
        AssertionError: If any repository containing the theorems has not been traced with LeanDojo.
    """
    repo, theorems, positions = _get_theorems_from_files(
        data_path,
        split,
        file_path,
        full_name,
        name_filter,
        num_theorems,
    )

    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    """
    Retrieves theorems from JSON files based on specified filters.

    This function loads theorem data from a JSON file and filters it based on optional parameters.
    It returns the repository information, a list of theorems, and their positions in the files.

    Args:
        data_path (str): Path to the directory containing the split JSON files.
        split (str): Name of the split file (without .json extension).
        file_path (Optional[str]): Filter theorems by specific file path.
        full_name (Optional[str]): Filter theorems by specific full name.
        name_filter (Optional[str]): Filter theorems by MD5 hash prefix of the full name.
        num_theorems (Optional[int]): Limit the number of theorems returned.

    Returns:
        Tuple[LeanGitRepo, List[Theorem], List[Pos]]: A tuple containing:
            - The Lean Git repository information
            - A list of Theorem objects
            - A list of Pos objects representing the positions of the theorems

    Note:
        Theorems are sorted by the MD5 hash of their file path and full name.
    """
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))

    # Jointly sort theorems and positions
    theorems_and_positions = list(zip(theorems, positions))
    theorems_and_positions.sort(
        key=lambda x: hashlib.md5(
            f"{x[0].file_path}:{x[0].full_name}".encode()
        ).hexdigest()
    )
    theorems, positions = zip(*theorems_and_positions)
    theorems, positions = list(theorems), list(positions)

    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    num_workers: int = 1,
    num_gpus: int = 0,
    verbose: bool = False,
) -> float:
    """
    Evaluates theorems using the distributed prover and returns pass rate.
    This function loads theorems from the specified data path, runs the distributed prover on them,
    and calculates statistics on how many theorems were successfully proved.
    Args:
        data_path (str): Path to the data repository containing theorems.
        exp_id (Optional[str]): Experiment ID to use for saving results. Defaults to a UUID if not provided.
        split (str): Dataset split to use ('val', 'test', etc.). Defaults to 'val'.
        file_path (Optional[str]): Path to a specific file to evaluate theorems from.
        full_name (Optional[str]): Full name of a specific theorem to evaluate.
        name_filter (Optional[str]): Filter theorems by name substring.
        num_theorems (Optional[int]): Maximum number of theorems to evaluate.
        ckpt_path (Optional[str]): Path to the checkpoint file for the prover model.
        indexed_corpus_path (Optional[str]): Path to the indexed corpus for the prover.
        tactic (Optional[str]): Specific tactic to use for proving.
        module (Optional[str]): Specific module to load for proving.
        num_sampled_tactics (int): Number of tactics to sample. Defaults to 64.
        timeout (int): Maximum time (in seconds) allowed for each theorem. Defaults to 600.
        num_workers (int): Number of worker processes to spawn. Defaults to 1.
        num_gpus (int): Number of GPUs to use. Defaults to 0.
        verbose (bool): Whether to output detailed logs. Defaults to False.
    Returns:
        float: The pass rate (num_proved / (num_proved + num_failed)) or NaN if no theorems were evaluated.
    """
    set_logger(verbose)

    repo, theorems, positions = _get_theorems(
        data_path, split, file_path, full_name, name_filter, num_theorems
    )

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        ckpt_path,
        indexed_corpus_path,
        tactic,
        module,
        num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
    )

    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    pickle_path = f"{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    """
    Main function for evaluating a theorem prover on theorems extracted by LeanDojo.

    The function parses command-line arguments and evaluates the prover using the specified configuration.
    It supports various filtering options for theorems, different tactic generators, and parallel proof search.

    Command-line arguments:
        --data-path: Path to the data extracted by LeanDojo
        --exp-id: Experiment ID used for logging
        --split: Dataset split to use (train, val, or test)
        --file-path: Filter theorems by file path
        --full-name: Filter theorems by full name
        --name-filter: Filter theorems by name pattern
        --num-theorems: Limit the number of theorems to evaluate
        --ckpt-path: Checkpoint path for the tactic generator model
        --indexed-corpus-path: Path to pickled indexed corpus for retrieval-based models
        --tactic: The tactic to evaluate (alternative to using a model)
        --module: The module to import the tactic from
        --num-sampled-tactics: Number of tactics to sample at each proof search node
        --timeout: Maximum seconds allowed for each proof search
        --num-workers: Number of concurrent provers for parallel evaluation
        --num-gpus: Number of GPUs to use for proof search
        --verbose: Enable detailed debugging output

    Returns:
        None: Results are logged with the final Pass@1 score
    """
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="The number of GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    args = parser.parse_args()

    assert args.ckpt_path or args.tactic
    assert args.num_gpus <= args.num_workers

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.num_workers,
        args.num_gpus,
        args.verbose,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
