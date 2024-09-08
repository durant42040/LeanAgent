"""Datamodule for the premise retrieval."""

import os
import json
import torch
import random
import itertools
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from lean_dojo import LeanGitRepo
from typing import Optional, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pickle


from common import Context, Corpus, Batch, Example, format_state, get_all_pos_premises


class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        corpus: Corpus,
        num_negatives: int,
        num_in_file_negatives: int,
        max_seq_len: int,
        tokenizer,
        is_train: bool,
        cache_path: str,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.cache_path = cache_path
        self.data = self.load_or_cache_data(data_paths)

    def load_or_cache_data(self, data_paths: List[str]) -> List[Example]:
        cache_file = os.path.join(self.cache_path, "cached_data.pkl")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                data = pickle.load(file)
            logger.info(f"Loaded data from cache {cache_file}")
        else:
            data = list(itertools.chain.from_iterable(self._load_data(path) for path in data_paths))
            # Cache the data
            # create file if it does not already exist
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass
            with open(cache_file, 'wb') as file:
                pickle.dump(data, file)
            logger.info(f"Saved loaded data to cache {cache_file}")
        return data

    def _load_data(self, data_path: str) -> List[Example]:
        data = []

        for thm in tqdm(json.load(open(data_path))):
            file_path = thm["file_path"]

            for i, tac in enumerate(thm["traced_tactics"]):
                state = format_state(tac["state_before"])
                # Some states are empty because they are from sorry theorems that have been proven.
                context = Context(
                    file_path, thm["full_name"], Pos(*thm["start"]), state if state else None
                )
                all_pos_premises = get_all_pos_premises(
                    tac["annotated_tactic"], self.corpus
                )

                if self.is_train:
                    # In training, we ignore tactics that do not have any premises.
                    for pos_premise in all_pos_premises:
                        data.append(
                            {
                                "url": thm["url"],
                                "commit": thm["commit"],
                                "file_path": thm["file_path"],
                                "full_name": thm["full_name"],
                                "start": thm["start"],
                                "tactic_idx": i,
                                "context": context,
                                "pos_premise": pos_premise,
                                "all_pos_premises": all_pos_premises,
                            }
                        )
                else:
                    data.append(
                        {
                            "url": thm["url"],
                            "commit": thm["commit"],
                            "file_path": thm["file_path"],
                            "full_name": thm["full_name"],
                            "start": thm["start"],
                            "tactic_idx": i,
                            "context": context,
                            "all_pos_premises": all_pos_premises,
                        }
                    )

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if not self.is_train:
            return self.data[idx]

        # In-file negatives + random negatives from all accessible premises.
        ex = deepcopy(self.data[idx])
        premises_in_file = []
        premises_outside_file = []

        for p in self.corpus.get_premises(ex["context"].path):
            if p == ex["pos_premise"]:
                continue
            if p.end < ex["context"].theorem_pos:
                if ex["pos_premise"].path == ex["context"].path:
                    premises_in_file.append(p)
                else:
                    premises_outside_file.append(p)

        for p in self.corpus.transitive_dep_graph.successors(ex["context"].path):
            if p == ex["pos_premise"].path:
                premises_in_file += [
                    _p for _p in self.corpus.get_premises(p) if _p != ex["pos_premise"]
                ]
            else:
                premises_outside_file += self.corpus.get_premises(p)

        num_in_file_negatives = min(len(premises_in_file), self.num_in_file_negatives)

        ex["neg_premises"] = random.sample(
            premises_in_file, num_in_file_negatives
        ) + random.sample(
            premises_outside_file, self.num_negatives - num_in_file_negatives
        )
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        # Tokenize the context.
        context = [ex["context"] for ex in examples]
        tokenized_context = self.tokenizer(
            [c.serialize() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask

        # Tokenize the label and premises.
        if self.is_train:
            pos_premise = [ex["pos_premise"] for ex in examples]
            tokenized_pos_premise = self.tokenizer(
                [p.serialize() for p in pos_premise],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            batch["pos_premise"] = pos_premise
            batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
            batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            # Check if one's negative is another's positive
            for j in range(batch_size):
                all_pos_premises = examples[j]["all_pos_premises"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        pos_premise_k = examples[k]["pos_premise"]
                    else:
                        pos_premise_k = examples[k % batch_size]["neg_premises"][
                            k // batch_size - 1
                        ]
                    label[j, k] = float(pos_premise_k in all_pos_premises)

            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.tokenizer(
                    [p.serialize() for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.corpus = Corpus(corpus_path)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = RetrievalDataset(
            [os.path.join(self.data_path, "train.json")],
            self.corpus,
            self.num_negatives,
            self.num_in_file_negatives,
            self.max_seq_len,
            self.tokenizer,
            is_train=True,
            cache_path=os.path.join(self.data_path, "cache_train")
        )
        print(f"Training dataset size: {len(self.ds_train)}")

        if stage in (None, "fit", "validate"):
            self.ds_val = RetrievalDataset(
                [os.path.join(self.data_path, "val.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
                cache_path=os.path.join(self.data_path, "cache_val")
            )
            print(f"Validation dataset size: {len(self.ds_val)}")

        if stage in (None, "fit", "predict"):
            self.ds_pred = RetrievalDataset(
                [os.path.join(self.data_path, "test.json")],
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train=False,
                cache_path=os.path.join(self.data_path, "cache_pred")
            )
            print(f"Testing dataset size: {len(self.ds_pred)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
