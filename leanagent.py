# import all the necessary libraries
import json
import os
import pickle
import random
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import ray
import torch
from lean_dojo import Theorem as LeanDojoTheorem
from lean_dojo import *
from loguru import logger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

import generate_benchmark_lean4
from database import DynamicDatabase, Theorem, AnnotatedTactic
from prover.proof_search import DistributedProver, SearchResult, Status
from retrieval.datamodule import RetrievalDataModule
from retrieval.main import run_cli
from retrieval.model import PremiseRetriever
from utils.constants import *
from utils.repository import should_skip_repo

# Set the seed for reproducibility
random.seed(3407)  # https://arxiv.org/abs/2109.08203

# Set up environment variables
os.environ["RAY_TMPDIR"] = f"{RAID_DIR}/tmp"

repos_for_merged_dataset = []
repos_for_proving = []

repos = []


PR_TITLE = "[LeanAgent] Proofs"

PR_BODY = """
[LeanAgent](https://arxiv.org/abs/2410.06209) discovers a proof for a theorem with the `sorry` keyword.

---

<i>~LeanAgent - From the [LeanDojo](https://leandojo.org/) family</i>
"""

TMP_BRANCH = "_LeanAgent"

COMMIT_MESSAGE = "[LeanAgent] Proofs"


def _eval(data, preds_map) -> Tuple[float, float, float]:
    """Evaluates the retrieval model."""
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = None
            key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            if key in preds_map:
                pred = preds_map[key]
            else:
                continue
            all_pos_premises = set(pred["all_pos_premises"])
            if len(all_pos_premises) == 0:
                continue

            retrieved_premises = pred["retrieved_premises"]
            TP1 = retrieved_premises[0] in all_pos_premises
            R1.append(float(TP1) / len(all_pos_premises))
            TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
            R10.append(float(TP10) / len(all_pos_premises))

            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    MRR.append(1.0 / (j + 1))
                    break
            else:
                MRR.append(0.0)

    R1 = 100 * np.mean(R1)
    R10 = 100 * np.mean(R10)
    MRR = np.mean(MRR)
    return R1, R10, MRR


def find_latest_checkpoint():
    """Finds the most recent checkpoint."""
    checkpoint_dir = RAID_DIR + "/" + CHECKPOINT_DIR
    all_checkpoints = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".ckpt")
    ]
    if not all_checkpoints:
        raise FileNotFoundError("No checkpoints found.")
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    logger.info(f"Using the latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def theorem_identifier(
    theorem: Theorem,
) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
    """Returns a unique identifier for a theorem."""
    return (
        theorem.full_name,
        str(theorem.file_path),
        tuple(theorem.start),
        tuple(theorem.end),
    )


def process_theorem_batch(
    theorem_batch, positions_batch, repo, db, prover, dynamic_database_json_path
):
    """Processes a batch of theorems."""
    lean_dojo_theorems = [t[1] for t in theorem_batch]
    results = prover.search_unordered(
        LeanGitRepo(repo.url, repo.commit), lean_dojo_theorems, positions_batch
    )

    # Create a mapping from LeanDojoTheorem to our Theorem
    theorem_map = {ldj_thm: thm for thm, ldj_thm in theorem_batch}

    for result in results:
        if not isinstance(result, SearchResult):
            logger.warning(f"Unexpected result type")
            continue

        if not result.theorem in theorem_map:
            logger.warning(f"Theorem not found in theorem_map: {result.theorem}")
            continue

        theorem = theorem_map[result.theorem]

        if not result.status == Status.PROVED:
            logger.info(f"No proof found for {theorem.full_name}")
            continue

        logger.info(f"Proof found for {theorem.full_name}")
        traced_tactics = [
            AnnotatedTactic(
                tactic=tactic,
                annotated_tactic=(tactic, []),
                state_before="",
                state_after="",
            )
            for tactic in result.proof
        ]
        theorem.traced_tactics = traced_tactics
        repo.change_sorry_to_proven(theorem, PROOF_LOG_FILE_NAME)
        db.update_repository(repo)
        logger.info(f"Updated theorem {theorem.full_name} in the database")

    db.to_json(dynamic_database_json_path)


def save_progress(all_encountered_theorems):
    """Saves the set of encountered theorems."""
    logger.info("Saving encountered theorems...")
    with open(ENCOUNTERED_THEOREMS_FILE, "wb") as f:
        pickle.dump(all_encountered_theorems, f)


def load_encountered_theorems(file_path):
    """Loads the theorems that have been encountered."""
    all_encountered_theorems = set()
    if not os.path.exists(file_path):
        logger.info(f"The file {file_path} does not exist. Starting with an empty set.")
        return set()

    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
            if file_content:  # Check if the file is not empty
                all_encountered_theorems = pickle.loads(file_content)
            else:
                logger.warning(
                    f"The file {file_path} is empty. Starting with an empty set."
                )
    except (EOFError, pickle.UnpicklingError) as e:
        logger.warning(f"Error reading {file_path}: {e}. Starting with an empty set.")
    except Exception as e:
        logger.error(
            f"Unexpected error when reading {file_path}: {e}. Starting with an empty set."
        )

    return all_encountered_theorems


def prove_sorry_theorems(
    db: DynamicDatabase,
    prover: DistributedProver,
    dynamic_database_json_path,
    repos_to_include: Optional[List[Tuple[str, str]]] = None,
    batch_size: int = 12,
):
    """Proves sorry theorems."""
    repos_to_process = (
        db.repositories
        if repos_to_include is None
        else [
            repo
            for repo in db.repositories
            if (repo.url, repo.commit) in repos_to_include
        ]
    )

    # To avoid proving the same theorem multiple times, potentially from different versions of the
    # same repo, we sort the repositories
    repos_to_process.sort(key=lambda r: r.metadata["date_processed"], reverse=True)

    processed_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = set()
    all_encountered_theorems: Set[Tuple[str, str, Tuple[int, int], Tuple[int, int]]] = (
        set()
    )
    last_save_time = datetime.datetime.now()
    save_interval = timedelta(minutes=30)

    # Load previously encountered theorems
    all_encountered_theorems = load_encountered_theorems(ENCOUNTERED_THEOREMS_FILE)

    for repo in repos_to_process:
        sorry_theorems = repo.sorry_theorems_unproved
        repo_url = repo.url
        repo_commit = repo.commit

        logger.info(f"Found {len(sorry_theorems)} sorry theorems to prove")

        theorem_batch = []
        positions_batch = []

        for theorem in tqdm(
            sorry_theorems, desc=f"Processing theorems from {repo.name}", unit="theorem"
        ):
            # Ignore sorry theorems from the repo's dependencies
            if theorem.url != repo_url or theorem.commit != repo_commit:
                continue

            theorem_id = theorem_identifier(theorem)

            if theorem_id in all_encountered_theorems:
                logger.info(
                    f"Skipping already encountered theorem: {theorem.full_name}"
                )
                continue

            all_encountered_theorems.add(theorem_id)
            if theorem_id in processed_theorems:
                logger.info(f"Skipping already processed theorem: {theorem.full_name}")
                continue

            processed_theorems.add(theorem_id)

            logger.info(f"Searching for proof for {theorem.full_name}")
            logger.info(f"Position: {theorem.start}")

            # Convert our Theorem to LeanDojo Theorem
            lean_dojo_theorem = LeanDojoTheorem(
                repo=LeanGitRepo(repo_url, repo_commit),
                file_path=theorem.file_path,
                full_name=theorem.full_name,
            )

            theorem_batch.append((theorem, lean_dojo_theorem))
            positions_batch.append(Pos(*theorem.start))

            if len(theorem_batch) == batch_size:
                process_theorem_batch(
                    theorem_batch,
                    positions_batch,
                    repo,
                    db,
                    prover,
                    dynamic_database_json_path,
                )
                theorem_batch = []
                positions_batch = []

            current_time = datetime.datetime.now()
            if current_time - last_save_time >= save_interval:
                save_progress(all_encountered_theorems)
                last_save_time = current_time

        # Process any remaining theorems in the last batch
        if theorem_batch:
            process_theorem_batch(
                theorem_batch,
                positions_batch,
                repo,
                db,
                prover,
                dynamic_database_json_path,
            )

    save_progress(all_encountered_theorems)
    logger.info("Finished attempting to prove sorry theorems")


def replace_sorry_with_proof(proofs):
    """Replace the `sorry` with the proof text in the Lean files."""
    logger.info(f"Replacing sorries with {len(proofs)} proofs!")
    # Group proofs by file paths
    proofs_by_file = {}
    for proof in proofs:
        file_path, start, end, proof_text, theorem_name = proof
        if file_path not in proofs_by_file:
            proofs_by_file[file_path] = []
        proofs_by_file[file_path].append((start, end, proof_text))

    for file_path, proofs in proofs_by_file.items():
        with open(file_path, "r") as file:
            lines = file.readlines()

        # sort proof by starting line and column number (working bottom up retains positions)
        proofs.sort(key=lambda x: (x[0].line_nb, x[0].column_nb), reverse=True)

        for start, end, proof_text in proofs:
            start_line, start_col = start.line_nb - 1, start.column_nb - 1
            end_line, end_col = end.line_nb - 1, end.column_nb - 1
            original_text = "".join(lines[start_line : end_line + 1])
            new_text = original_text.replace("sorry", proof_text, 1)
            lines[start_line : end_line + 1] = new_text

            with open(file_path, "w") as file:
                file.writelines(lines)

    logger.info("Finished replacing sorries with proofs!")


def main():
    """
    Main function to run LeanAgent.
    """
    global repos_for_merged_dataset
    global repos_for_proving
    try:
        current_epoch = 0
        epochs_per_repo = 1
        run_progressive_training = True
        single_repo = True
        curriculum_learning = True
        num_repos = 1
        dynamic_database_json_path = RAID_DIR + "/" + DB_FILE_NAME

        # Check if the current process is the main one
        is_main_process = int(os.environ.get("LOCAL_RANK", "0")) == 0
        # Initialize database and discover repositories
        if is_main_process:
            db = DynamicDatabase(json_path=dynamic_database_json_path)
            lean_git_repos = db.setup_repositories(
                num_repos,
                curriculum_learning,
            )

        lambdas = None
        if run_progressive_training:
            logger.info("Running progressive training")
            lambdas = [0.1]
        else:
            logger.info("Running retrieval baseline")
            lambdas = [0.0]
        # Iterate over each repository and lambda value
        for i in range(num_repos):
            for lambda_value in lambdas:
                logger.info(f"length of lean_git_repos: {len(lean_git_repos)}")
                logger.info(f"i: {i}")
                repo = lean_git_repos[i]
                sha = repo.commit
                dir_name = repo.url.split("/")[-1] + "_" + sha
                result = True
                if is_main_process:
                    logger.info("Main process")
                    logger.info(f"Using lambda = {lambda_value}")
                    logger.info(f"Processing {repo.url}")

                    if single_repo:
                        repos_for_merged_dataset = []
                    repos_for_proving = []

                    # Create a directory for the merged dataset if it doesn't exist
                    dst_dir = Path(RAID_DIR) / DATA_DIR / f"merged_with_new_{dir_name}"
                    if (repo.url, repo.commit) not in repos_for_merged_dataset:
                        logger.info("Adding repo to repos_for_merged_dataset")
                        repos_for_merged_dataset.append((repo.url, repo.commit))
                        if not single_repo:
                            repos_for_proving.append((repo.url, repo.commit))
                    else:
                        logger.info("Repo already in repos_for_merged_dataset")

                    db.generate_merged_dataset(dst_dir, repos_for_merged_dataset)

                dst_dir = (
                    RAID_DIR + "/" + DATA_DIR + "/" + f"merged_with_new_{dir_name}"
                )
                new_data_path = dst_dir

                logger.info("All GPUs")
                model_checkpoint_path = None
                best_model = None
                data_module = None
                if run_progressive_training:
                    try:
                        model_checkpoint_path = find_latest_checkpoint()
                        logger.info(f"Found latest checkpoint: {model_checkpoint_path}")
                    except FileNotFoundError as e:
                        logger.error(str(e))
                        model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"

                    # Train the model on the new dataset that we generated from the dynamic database.
                    logger.info("Inside train_test_fisher")
                    logger.info(f"Starting training at epoch {current_epoch}")
                    seed_everything(3407)

                    # Progessive Training

                    if not torch.cuda.is_available():
                        logger.warning(
                            "Indexing the corpus using CPU can be very slow."
                        )
                        device = torch.device("cpu")
                    else:
                        device = torch.device("cuda")

                    config = {
                        "model_name": "kaiyuy/leandojo-lean4-retriever-byt5-small",
                        "lr": 1e-3,
                        "warmup_steps": 1000,
                        "max_seq_len": 512,
                        "num_retrieved": 100,
                    }

                    model = PremiseRetriever.load(
                        model_checkpoint_path, device, freeze=False, config=config
                    )
                    model.train()
                    logger.info(f"Loaded premise retriever at {model_checkpoint_path}")

                    # Initialize ModelCheckpoint and EarlyStopping
                    dir_name = new_data_path.split("/")[-1]
                    filename_suffix = f"_lambda_{lambda_value}"
                    checkpoint_callback = ModelCheckpoint(
                        dirpath=RAID_DIR + "/" + CHECKPOINT_DIR,
                        filename=dir_name
                        + filename_suffix
                        + "_{epoch}-{Recall@10_val:.2f}",
                        verbose=True,
                        save_top_k=-1,  # Save all checkpoints
                        every_n_epochs=1,  # Save every epoch (which is just once in this case)
                        monitor="Recall@10_val",
                        mode="max",
                    )

                    early_stop_callback = EarlyStopping(
                        monitor="Recall@10_val", patience=5, mode="max", verbose=True
                    )

                    lr_monitor = LearningRateMonitor(logging_interval="step")

                    # Set up environment variables for NCCL
                    VERY_LONG_TIMEOUT = 7 * 24 * 60 * 60 * 52  # 1 year
                    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
                    os.environ["NCCL_TIMEOUT"] = str(VERY_LONG_TIMEOUT * 1000)

                    # Create a custom log directory for Lightning
                    custom_log_dir = os.path.join(
                        RAID_DIR,
                        "lightning_logs",
                        f"{dir_name}_{False}_lambda_{lambda_value}",
                    )
                    os.makedirs(custom_log_dir, exist_ok=True)

                    # Initialize DDP strategy
                    ddp_strategy = DDPStrategy(
                        timeout=timedelta(seconds=VERY_LONG_TIMEOUT)
                    )
                    trainer = pl.Trainer(
                        accelerator="gpu",
                        gradient_clip_val=1.0,
                        precision="bf16-mixed",
                        strategy=ddp_strategy,
                        devices=4,
                        accumulate_grad_batches=4,
                        callbacks=[
                            lr_monitor,
                            checkpoint_callback,
                            early_stop_callback,
                        ],
                        max_epochs=current_epoch + epochs_per_repo,
                        log_every_n_steps=1,
                        num_sanity_val_steps=0,
                        default_root_dir=custom_log_dir,
                    )

                    # Barrier before data module
                    logger.info("right before barrier for data module")
                    trainer.strategy.barrier()
                    should_skip, skip_repo_url = should_skip_repo()
                    if should_skip:
                        logger.info(
                            f"Skipping repository {skip_repo_url} due to preprocessing issues"
                        )
                        trainer.strategy.barrier()
                        if is_main_process:
                            logger.info("Removing skip file")
                            skip_file_path = os.path.join(
                                RAID_DIR, DATA_DIR, "skip_repo.txt"
                            )
                            os.remove(skip_file_path)
                        continue

                    # Set lambda value for the model
                    model.set_lambda(lambda_value)
                    corpus_path = new_data_path + "/corpus.jsonl"
                    data_path = new_data_path + "/random"
                    logger.info(f"Data path: {data_path}")
                    data_module = RetrievalDataModule(
                        data_path=data_path,
                        corpus_path=corpus_path,
                        num_negatives=3,
                        num_in_file_negatives=1,
                        model_name="google/byt5-small",
                        batch_size=BATCH_SIZE,
                        eval_batch_size=64,
                        max_seq_len=1024,
                        num_workers=4,
                    )
                    data_module.setup(stage="fit")

                    logger.info(
                        f"Training dataset size after load: {len(data_module.ds_train)}"
                    )
                    logger.info(
                        f"Validation dataset size after load: {len(data_module.ds_val)}"
                    )
                    logger.info(
                        f"Testing dataset size after load: {len(data_module.ds_pred)}"
                    )

                    logger.info(
                        f"Starting progressive training from epoch {current_epoch} to {current_epoch + epochs_per_repo}"
                    )

                    # Train the model
                    try:
                        logger.info("hit the barrier before training")
                        trainer.strategy.barrier()
                        trainer.fit(
                            model,
                            datamodule=data_module,
                            ckpt_path=model_checkpoint_path,
                        )
                        logger.info("hit the barrier after training")
                        trainer.strategy.barrier()
                    except Exception as e:
                        print(f"An error occurred during training: {str(e)}")
                        print(traceback.format_exc())

                    logger.info(
                        f"Finished progressive training at epoch {trainer.current_epoch}"
                    )

                    # Testing for Average Recall

                    try:
                        best_model_path = find_latest_checkpoint()
                        logger.info(f"Found latest checkpoint: {best_model_path}")
                        best_model = PremiseRetriever.load(
                            best_model_path, device, freeze=False, config=config
                        )
                    except FileNotFoundError as e:
                        logger.error(f"No checkpoint found: {str(e)}")
                        logger.warning("Using the current model state.")
                        best_model = model

                    best_model.eval()

                    logger.info("Testing...")
                    total_R1, total_R10, total_MRR = [], [], []
                    dataset_path = RAID_DIR + "/" + DATA_DIR
                    testing_paths = [
                        os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                    ]
                    if is_main_process:
                        with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                            f.write("\n\n\n")
                            f.write(
                                f"Results for {dir_name} with lambda = {lambda_value}"
                            )
                    for data_path in testing_paths:
                        if "merged" not in data_path:
                            continue

                        run_cli(best_model_path, data_path)
                        if is_main_process:
                            num_gpus = 4
                            preds_map = {}
                            for gpu_id in range(num_gpus):
                                with open(f"test_pickle_{gpu_id}.pkl", "rb") as f:
                                    preds = pickle.load(f)
                                    preds_map.update(preds)

                            logger.info("Loaded the predictions pickle files")
                            data_path = os.path.join(data_path, "random", "test.json")
                            data = json.load(open(data_path))
                            logger.info(f"Evaluating on {data_path}")
                            R1, R10, MRR = _eval(data, preds_map)
                            logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")
                            total_R1.append(R1)
                            total_R10.append(R10)
                            total_MRR.append(MRR)
                            with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                                f.write("\n\n\n")
                                f.write(f"Intermediate results for {data_path}")
                                f.write("\n\n\n")
                                f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

                    if is_main_process:
                        avg_R1 = np.mean(total_R1)
                        avg_R10 = np.mean(total_R10)
                        avg_MRR = np.mean(total_MRR)

                        logger.info(
                            f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}"
                        )

                        if not os.path.exists(EVAL_RESULTS_FILE_PATH):
                            open(EVAL_RESULTS_FILE_PATH, "w").close()

                        with open(EVAL_RESULTS_FILE_PATH, "a") as f:
                            f.write("\n\n\n")
                            f.write(
                                f"Average R@1 = {avg_R1} %, R@10 = {avg_R10} %, MRR = {avg_MRR}"
                            )
                else:
                    model_checkpoint_path = f"{RAID_DIR}/checkpoints/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5.ckpt"
                    if result is None:
                        logger.info(
                            f"Skipping repository {repo.url} due to preprocessing issues"
                        )
                        continue

                if is_main_process:
                    logger.info("Starting the prover")

                    if ray.is_initialized():
                        logger.info("Shutting down Ray before proving")
                        ray.shutdown()

                    # Set up the prover
                    use_vllm = False
                    corpus_path = dst_dir + "/corpus.jsonl"
                    tactic = (
                        None  # `None` since we are not using a fixed tactic generator
                    )
                    module = (
                        None  # `None` since we are not using a fixed tactic generator
                    )
                    num_workers = 4
                    num_gpus = 4
                    timeout = 600
                    max_expansions = None
                    num_sampled_tactics = 64
                    debug = False
                    ckpt_path = f"{RAID_DIR}/model_lightning.ckpt"
                    prover = DistributedProver(
                        use_vllm,
                        ckpt_path,
                        corpus_path,
                        tactic,
                        module,
                        num_workers,
                        num_gpus=num_gpus,
                        timeout=timeout,
                        max_expansions=max_expansions,
                        num_sampled_tactics=num_sampled_tactics,
                        raid_dir=RAID_DIR,
                        checkpoint_dir=CHECKPOINT_DIR,
                        debug=debug,
                        run_progressive_training=run_progressive_training,
                    )

                    # Prove sorry theorems
                    if single_repo:
                        prove_sorry_theorems(
                            db,
                            prover,
                            dynamic_database_json_path,
                            repos_for_merged_dataset,
                        )
                    else:
                        prove_sorry_theorems(
                            db, prover, dynamic_database_json_path, repos_for_proving
                        )
                    db.to_json()

                    logger.info("Finished searching for proofs of sorry theorems")

                    if ray.is_initialized():
                        logger.info("Shutting down Ray after proving")
                        ray.shutdown()

                logger.info("Finished processing the repository")
                current_epoch += epochs_per_repo
                logger.info(f"current epoch: {current_epoch}")

    except Exception as e:
        logger.info(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()


if __name__ == "__main__":
    main()
