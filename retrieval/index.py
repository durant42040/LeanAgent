"""Script for indexing the corpus using the retriever.
"""

import argparse
import pickle

import torch
from loguru import logger

from common import IndexedCorpus
from retrieval.model import PremiseRetriever


def main() -> None:
    """
    Main function for training and saving a BM25 premise retriever.

    This function parses command line arguments, initializes a PremiseRetriever model,
    loads a corpus, reindexes it, and saves the indexed corpus to a specified output path.

    Command line arguments:
        --ckpt_path: Path to the checkpoint for loading the PremiseRetriever model.
        --corpus-path: Path to the corpus file to be indexed.
        --output-path: Path where the indexed corpus will be saved.
        --batch-size: Batch size for corpus indexing (default: 64).

    The function will use CUDA if available, with a warning that CPU indexing can be slow.
    """
    parser = argparse.ArgumentParser(
        description="Script for training the BM25 premise retriever."
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--corpus-path", type=str, required=True)
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    logger.info(args)

    if not torch.cuda.is_available():
        logger.warning("Indexing the corpus using CPU can be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    model = PremiseRetriever.load_hf(args.ckpt_path, 2048, device)
    model.load_corpus(args.corpus_path)
    model.reindex_corpus(batch_size=args.batch_size)

    pickle.dump(
        IndexedCorpus(model.corpus, model.corpus_embeddings.to(torch.float32).cpu()),
        open(args.output_path, "wb"),
    )
    logger.info(f"Indexed corpus saved to {args.output_path}")


if __name__ == "__main__":
    main()

# python retrieval/index.py --ckpt_path leandojo-lean4-retriever-byt5-small --corpus-path /data/yingzi_ma/lean_project/datasets/mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5/corpus.jsonl --output-path indexed_corpus.pkl
# python retrieval/index.py --ckpt_path leandojo-lean4-retriever-byt5-small --corpus-path mathlib4_29dcec074de168ac2bf835a77ef68bbe069194c5/corpus.jsonl --output-path indexed_corpus.pkl
