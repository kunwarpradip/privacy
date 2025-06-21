import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from TTLoRAWrapper_utils import wrap_model_with_ttcores_contraction
from utils import load_new_model_for_sequence_classification_from_local_path, wrap_model_with_lora
import torchmetrics
from pytorch_lightning.callbacks import Callback
import time
import psutil
import bitsandbytes as bnb
from opacus import PrivacyEngine


class DPSequenceClassifier(pl.LightningModule):
    def __init__(self, config, train_loader, val_loader):
        super().__init__()
        self.epoch_times = []
        self.epoch_memories = []
        self.epoch_gpu_peak = []

        self.save_hyperparameters(config)

        self.privacy_engine = PrivacyEngine(accountant="rdp")

        self._train_loader = train_loader
        self._val_loader = val_loader
        self.model = load_new_model_for_sequence_classification_from_local_path(self.hparams)
        if self.hparams['fttype'] == "ttlora":
            self.model = wrap_model_with_ttcores_contraction(self.model, self.hparams)
        elif self.hparams['fttype'] == "lora":
            self.model = wrap_model_with_lora(self.model, self.hparams)
        elif self.hparams['fttype'] == "fft":
             pass

        self.data = config["dataset_name"]
        if self.data == "socialiqa" or self.data =="sick" or self.data == "cb"in self.data or self.data == "mnli":
            self.num_classes = 3
        elif self.data == "cosmosqa" or self.data == "hellaswag":
            self.num_classes = 4
        elif self.data == "csqa":
            self.num_classes = 5
        else:
            self.num_classes = 2

        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def configure_optimizers(self):
        optimizer = bnb.optim.AdamW8bit(self.parameters(), lr=self.hparams.learning_rate)
        self.model, optimizer, _, data_loader = self.privacy_engine.make_private_with_epsilon( #_ for criterion as opacus uses its own
            module=self.model,
            optimizer=optimizer,
            data_loader=self._train_loader,
            target_delta=self.hparams['delta'],
            target_epsilon=self.hparams['epsilon'],
            epochs=self.hparams['epochs'],
            max_grad_norm=self.hparams['max_grad_norm'],
            grad_sample_mode="ghost",
            )
        self._wrapped_train_loader = data_loader
        return optimizer
    
    def train_dataloader(self):
        return self._wrapped_train_loader
    
    def val_dataloader(self):
        return self._val_loader
    
    def on_train_epoch_end(self):
        epsilon = self.privacy_engine.get_epsilon(delta=self.hparams['delta'])
        self.log("epsilon_epoch", epsilon, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        loss = outputs.loss

        preds = torch.argmax(outputs.logits, dim=1)
        self.train_acc(preds, batch["label"])
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return outputs["loss"]  # this is passed to the optimizer for training
    
    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
        self.log("val_loss", outputs["loss"], prog_bar=False)
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("val_acc", self.val_acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["label"])
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("accuracy", self.test_acc, prog_bar=True)

    def on_before_zero_grad(self, optimizer):
        """
        This hook is called after every optimizer.step(). We add a condition
        to only execute the logic on the very last training batch of an epoch.
        """
        # The 'is_last_batch' property is True only for the last training step of an epoch.
        # We also check for global_rank == 0 to print only from the main process in DDP.
        if self.trainer.is_last_batch and self.global_rank == 0:
            print(f"\n--- Final Private Gradients for Epoch {self.current_epoch} (from last batch) ---")
            
            # Iterate through all named parameters of the model
            for name, param in self.model.named_parameters():
                
                # Check if the parameter is trainable and has a gradient
                if param.requires_grad and param.grad is not None:
                    
                    # Print summary statistics of the final private gradient for this epoch
                    print(
                        f"Layer: {name:<40} | "
                        f"Final Grad Norm: {param.grad.norm():.6f} | "
                        f"Final Grad Mean: {param.grad.mean():.6f} | \n"
                        # f"Grad value     : \n{param.grad}"
                    )
            print("--------------------------------------------------------------------------\n")

class TimeMemoryCallback(Callback):
   def __init__(self):
       super().__init__()
       self.epoch_times = []
       self.epoch_memories = []
       self.epoch_gpu_peak = []

   def on_train_epoch_start(self, trainer, pl_module):
       # Record the start time of this epoch
       torch.cuda.reset_peak_memory_stats(device=pl_module.device)
       self._start_time = time.time()

   def on_train_epoch_end(self, trainer, pl_module):
       # Compute elapsed time for this epoch
       elapsed = time.time() - self._start_time
       self.epoch_times.append(elapsed)
      
       # get peak GPU memory (in bytes) since last reset
       peak_bytes = torch.cuda.max_memory_allocated(device=pl_module.device)
       peak_mb = peak_bytes / (1024 ** 2)
       self.epoch_gpu_peak.append(peak_mb)

       # CPU RSS (in MB)
       mem_mb = psutil.Process().memory_info().rss / (1024 ** 2)
       self.epoch_memories.append(mem_mb)











