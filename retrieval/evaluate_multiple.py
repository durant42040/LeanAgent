import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm


def _eval(data, preds_map) -> Tuple[float, float, float]:
    """
    Evaluates the performance of premise retrieval against ground truth.

    Parameters:
    -----------
    data : list
        List of theorem data dictionaries, where each dictionary contains information about
        theorems including file_path, full_name, start position, and traced_tactics.
    preds_map : dict
        Dictionary mapping theorem identifiers to prediction results. Each prediction contains
        'all_pos_premises' (ground truth) and 'retrieved_premises' (model predictions).

    Returns:
    --------
    Tuple[float, float, float]
        A tuple containing three evaluation metrics:
        - R1: Top-1 Recall (percentage of times the top retrieved premise is a correct premise)
        - R10: Recall@10 (percentage of correct premises found in the top 10 retrievals)
        - MRR: Mean Reciprocal Rank (average of 1/rank where rank is the position of the first correct premise)

    Notes:
    ------
    - For each tactic in each theorem, the function evaluates if the retrieved premises match the ground truth.
    - If a theorem/tactic is not in preds_map or has no positive premises, it is skipped.
    - All metrics are normalized: R1 and R10 are percentages (0-100), MRR is between 0 and 1.
    """
    R1, R10, MRR = [], [], []
    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = preds_map.get(
                (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            )
            if pred is None:
                continue
            all_pos_premises = set(pred["all_pos_premises"])
            if not all_pos_premises:
                continue

            retrieved_premises = pred["retrieved_premises"]
            TP1 = retrieved_premises[0] in all_pos_premises
            R1.append(float(TP1))
            TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
            R10.append(float(TP10) / len(all_pos_premises))

            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    MRR.append(1.0 / (j + 1))
                    break
            else:
                MRR.append(0.0)

    R1 = 100 * np.mean(R1) if R1 else 0
    R10 = 100 * np.mean(R10) if R10 else 0
    MRR = np.mean(MRR) if MRR else 0
    return R1, R10, MRR


def main():
    """
    Main function for evaluating the premise retriever on multiple data splits.

    The function loads the predictions from a file, evaluates them against each provided data split,
    and calculates average metrics across all splits. Metrics include Recall@1, Recall@10, and Mean
    Reciprocal Rank (MRR).

    Command Line Arguments:
        --preds-file (str): Path to the pickle file containing retriever's predictions.
        --data-paths (list of str): Paths to JSON files containing data splits for evaluation.

    Returns:
        None: Results are logged to the console.
    """
    parser = argparse.ArgumentParser(
        description="Script for evaluating the premise retriever."
    )
    parser.add_argument(
        "--preds-file",
        type=str,
        required=True,
        help="Path to the retriever's predictions file.",
    )
    parser.add_argument(
        "--data-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing the data splits.",
    )

    args = parser.parse_args()
    logger.info(f"Loading predictions from {args.preds_file}")
    preds = pickle.load(open(args.preds_file, "rb"))
    preds_map = {
        (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
        for p in preds
    }

    total_R1, total_R10, total_MRR = [], [], []
    for data_path in args.data_paths:
        logger.info(f"Evaluating on {data_path}")
        data = json.load(open(data_path))
        R1, R10, MRR = _eval(data, preds_map)
        total_R1.append(R1)
        total_R10.append(R10)
        total_MRR.append(MRR)
        logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

    avg_R1 = np.mean(total_R1)
    avg_R10 = np.mean(total_R10)
    avg_MRR = np.mean(total_MRR)
    logger.info(
        f"Average R@1 = {avg_R1} %, Average R@10 = {avg_R10} %, Average MRR = {avg_MRR}"
    )


if __name__ == "__main__":
    main()
