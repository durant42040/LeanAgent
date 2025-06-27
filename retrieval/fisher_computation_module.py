import pickle

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from loguru import logger


class FisherComputationModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fisher_info = {}
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step within a PyTorch Lightning module.

        This method:
        1. Resets gradients of the model
        2. Computes the loss using the model on the given batch
        3. Performs manual backward pass for the loss
        4. Updates Fisher information matrix for each parameter that has gradients

        Parameters:
        ----------
        batch : dict
            Dictionary containing batch data with keys:
            - context_ids: Tensor of context token IDs
            - context_mask: Attention mask for context
            - pos_premise_ids: Tensor of positive premise token IDs
            - pos_premise_mask: Attention mask for positive premises
            - neg_premises_ids: Tensor of negative premises token IDs
            - neg_premises_mask: Attention mask for negative premises
            - label: Tensor of labels for the batch

        batch_idx : int
            Index of the batch in the current epoch

        Returns:
        -------
        torch.Tensor
            The computed loss for this batch

        Notes:
        -----
        This method accumulates the squared gradients in the Fisher information dictionary,
        which can be used for techniques like Elastic Weight Consolidation (EWC).
        """
        self.model.zero_grad()
        loss = self.model(
            batch["context_ids"],
            batch["context_mask"],
            batch["pos_premise_ids"],
            batch["pos_premise_mask"],
            batch["neg_premises_ids"],
            batch["neg_premises_mask"],
            batch["label"],
        )

        self.manual_backward(loss)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if name not in self.fisher_info:
                    self.fisher_info[name] = param.grad.detach().clone() ** 2
                else:
                    self.fisher_info[name] += param.grad.detach().clone() ** 2

        return loss

    def on_train_epoch_end(self):
        """
        Synchronizes and normalizes Fisher Information across all GPUs at the end of each training epoch.
        This method performs two main operations:
        1. Synchronizes Fisher Information by summing values from all GPUs using all_reduce operation
        2. Normalizes the Fisher Information by dividing by the total number of samples processed
           (dataset size * number of GPUs)
        The resulting Fisher Information values represent the average importance of each parameter
        across the entire distributed training process.
        """
        logger.info("Synchronizing and normalizing Fisher Information")

        # Synchronize Fisher Information across GPUs
        # Each GPU now has the sum of the Fisher Information from all GPUs for each parameter
        for name in self.fisher_info:
            dist.all_reduce(self.fisher_info[name], op=dist.ReduceOp.SUM)

        logger.info("Normalizing Fisher Information")
        logger.info(f"length of dataset: {len(self.trainer.train_dataloader.dataset)}")
        world_size = dist.get_world_size()
        logger.info(f"world_size: {world_size}")
        dataset_size = len(self.trainer.train_dataloader.dataset)
        logger.info(f"dataset_size: {dataset_size}")
        total_samples = dataset_size * world_size
        logger.info(f"total_samples: {total_samples}")
        for name in self.fisher_info:
            self.fisher_info[name] /= total_samples

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(), lr=0
        )  # We don't actually want to update the model

    def save_fisher_info(self, fisher_file_path):
        if self.trainer.is_global_zero:
            logger.info(f"Saving Fisher Information Matrix to {fisher_file_path}")
            with open(fisher_file_path, "wb") as f:
                pickle.dump(self.fisher_info, f)
