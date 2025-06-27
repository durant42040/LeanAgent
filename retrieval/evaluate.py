"""Script for evaluating the premise retriever."""

import argparse
import json
import os
import pickle
from typing import Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm


def _eval(data, preds_map) -> Tuple[float, float, float]:
    """
    Evaluates the performance of a retrieval system for theorem premises.

    This function computes three evaluation metrics:
    - R1: Recall at 1, measuring the percentage of positive premises correctly retrieved as the top result
    - R10: Recall at 10, measuring the percentage of positive premises correctly retrieved within the top 10 results
    - MRR: Mean Reciprocal Rank, measuring the average of reciprocal ranks of the first relevant premise

    Parameters
    ----------
    data : list
        List of theorem dictionaries, each containing 'file_path', 'full_name', 'start', and 'traced_tactics'
    preds_map : dict
        Dictionary mapping theorem identifiers (file_path, full_name, start, index) to prediction results,
        where each prediction result has 'all_pos_premises' and 'retrieved_premises' fields

    Returns
    -------
    Tuple[float, float, float]
        A tuple containing (R1, R10, MRR) scores, where R1 and R10 are percentages (0-100)
        and MRR is a value between 0 and 1
    """
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            pred = preds_map[
                (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            ]
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


def main() -> None:
    """
    Main function for evaluating premise retriever performance.

    This script loads predictions from a file and test data, then calculates and logs
    evaluation metrics (R@1, R@10, MRR). Results are both logged and saved to a text file.

    Command-line arguments:
        --preds-file: Path to the retriever's predictions file (pickle format)
        --data-path: Path to the directory containing the train/val/test splits

    Example usage:
        python evaluate.py --preds-file predictions.pkl --data-path data/
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
        "--data-path",
        type=str,
        required=True,
        help="Path to the directory containing the train/val/test splits.",
    )
    args = parser.parse_args()
    logger.info(args)

    logger.info(f"Loading predictions from {args.preds_file}")
    preds = pickle.load(open(args.preds_file, "rb"))
    preds_map = {
        (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
        for p in preds
    }
    assert len(preds) == len(preds_map), "Duplicate predictions found!"

    data_path = os.path.join(args.data_path, "test.json")
    data = json.load(open(data_path))
    logger.info(f"Evaluating on {data_path}")
    R1, R10, MRR = _eval(data, preds_map)
    logger.info(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")

    file_path = "BM25_downloaded_evaluation_results.txt"
    with open(file_path, "w") as f:
        f.write(f"R@1 = {R1} %, R@10 = {R10} %, MRR = {MRR}")


if __name__ == "__main__":
    main()
