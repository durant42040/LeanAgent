import argparse
import os

from loguru import logger
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from common import Corpus
from retrieval.datamodule import RetrievalDataset


def main() -> None:
    """
    Train a BPE tokenizer using premises from a corpus and states from a retrieval dataset.
    
    This function:
    1. Parses command line arguments for vocabulary size, data path, and output path
    2. Initializes a BPE tokenizer with special tokens and whitespace pre-tokenizer
    3. Loads premises from a corpus and states from a retrieval dataset
    4. Trains the tokenizer on both premises and states
    5. Saves the trained tokenizer to the specified output path
    
    Command Line Arguments:
        --vocab-size: Size of the vocabulary (default: 30000)
        --data-path: Path to the input data directory (required)
        --output-path: Path to save the trained tokenizer (required)
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    logger.info(args)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    corpus = Corpus(os.path.join(args.data_path, "../corpus.jsonl"))
    premises = [premise.serialize() for premise in corpus.all_premises]

    ds_train = RetrievalDataset(
        [os.path.join(args.data_path, "train.json")],
        corpus,
        num_negatives=0,
        num_in_file_negatives=0,
        max_seq_len=1024,
        tokenizer=None,
        is_train=False,
        cache_path=os.path.join(args.data_path, "cache_train")
    )
    states = [
        ds_train.data[i]["context"].serialize() for i in range(len(ds_train.data))
    ]
    tokenizer.train_from_iterator(premises + states, trainer=trainer)
    tokenizer.save(args.output_path)
    logger.info(f"Tokenizer saved to {args.output_path}")


if __name__ == "__main__":
    main()
