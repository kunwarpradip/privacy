import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import CSVLogger
from utils import get_tokenizer, load_dataset_, preprocess_datasets
from Lightning_Model import DPSequenceClassifier, TimeMemoryCallback
from opacus import PrivacyEngine
import sys
import warnings
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
import pandas as pd
torch.cuda.reset_peak_memory_stats()

parser = ArgumentParser()
parser.add_argument("--epsilon", type=float, default=3.0)
parser.add_argument("--batchsize", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--dataset", type=str, default="rte")
parser.add_argument("--fttype", type=str, default="ttlora")
args = parser.parse_args()

config = {
  "model_path": "/lustre/scratch5/pkunwar/llms/llama3.2-1b/checkpoints",
  "tokenizer_path": "/lustre/scratch5/pkunwar/llms/llama3.2-1b/checkpoints",
  "dataset_path": "/lustre/scratch5/pkunwar/datasets",
  "dataset_name": args.dataset,
  "model_name": "llama3.2-1b",
  "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
 
  "learning_rate": 5e-3 if args.fttype=="ttlora" or args.fttype=="fft" else 5e-4,
  "epochs": args.epochs,
  "patience" : args.patience,
  "batchsize" : args.batchsize,
  #private training parameters
  "delta": 1e-5,
  "epsilon": args.epsilon,
  "max_grad_norm": 1.0,
  # TTLoRA parameters
  "qshape": [16, 8, 4, 4, 4, 4, 8, 16],
  "m_factors_q": [16, 8, 4, 4],
  "n_factors_q": [16, 8, 4, 4],
  "vshape": [16, 16, 4, 2, 2, 16, 16],
  "m_factors_v": [16, 16, 4, 2],
  "n_factors_v": [16, 16, 2],
  "rank": 5,
  "alpha": 16,
  "core_init_choice": "direct_init",

  "lora_rank" : 16,
  "lora_alpha": 8,

  "fttype": args.fttype,
}

raw_dataset = load_dataset_(config)
dataset = preprocess_datasets(config, raw_dataset)
tokenized = get_tokenizer(config, dataset)
keep_columns = ['label', 'input_ids', 'attention_mask']
for split in tokenized.keys():
  remove_cols = [col for col in dataset[split].column_names if col not in keep_columns]
  tokenized[split] = tokenized[split].remove_columns(remove_cols)

train_dataset = tokenized["train"]
val_dataset = tokenized["validation"]

train_loader = DataLoader(train_dataset, batch_size=config["batchsize"], shuffle=False, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=config["batchsize"], num_workers=8)

model = DPSequenceClassifier(config, train_loader, val_loader)
print(model)
model.train()

print(f"Opacus PrivacyEngine attached. Target Epsilon: {config['epsilon']}")

early_stopping_callback = EarlyStopping(
      monitor='val_loss',
      patience=config["patience"],
      verbose=True,
      mode='min'
      )

model_checkpoint_callback=ModelCheckpoint(
      dirpath=f'./{config["fttype"]}_DP_Experts/{config["dataset_name"]}/{config["epsilon"]}',
      save_top_k=1,
      mode="max",
      monitor="val_acc")

time_mem_cb = TimeMemoryCallback()
# privacy_cb = PrivacyMetricsCallback(privacy_engine, config['delta'])

trainer = Trainer(
  max_epochs=config["epochs"],
  accelerator="gpu",
#    strategy=DDPStrategy(find_unused_parameters=True),
  logger=CSVLogger("logs", name="dp_ttlora"),
  devices=torch.cuda.device_count(),
  # devices=1,
  use_distributed_sampler=False,
  callbacks = [time_mem_cb, early_stopping_callback, model_checkpoint_callback],
  # precision="16-mixed",
)

trainer.fit(model=model)
final_epsilon = model.privacy_engine.accountant.get_epsilon(delta=config['delta'])
print("Final Epsilon", final_epsilon)
print(f"Training finished. Final Epsilon: {final_epsilon}")
avg_time = sum(time_mem_cb.epoch_times) / len(time_mem_cb.epoch_times)
avg_mem = sum(time_mem_cb.epoch_memories) / len(time_mem_cb.epoch_memories)
avg_gpu_mem = sum(time_mem_cb.epoch_gpu_peak) / len(time_mem_cb.epoch_gpu_peak)
print(f"→ Average time/epoch: {avg_time:.2f} sec")
print(f"→ Average memory (RSS)/epoch: {avg_mem:.2f} MB\n")
print(f"→ Average GPU memory/epoch: {avg_gpu_mem:.2f} MB\n")

'''Evaluating the model on training and validation datasets'''
train_acc = trainer.test(model, dataloaders=train_loader, ckpt_path="best", verbose=False)
val_acc = trainer.test(model, dataloaders=val_loader, ckpt_path="best", verbose=False)
print("-" * 50,
     "\nTraining Accuracy: ", train_acc,
     "\nValidation Accuracy in best lightning model: ", val_acc)

print("Best model path: ", model_checkpoint_callback.best_model_path)

def count_parameters(model):
   return sum(p.numel() for p in model.parameters() if p.requires_grad)
train_params = count_parameters(model)

print(f"Total trainable parameters: {train_params}")

results = {
    "Fine Tuning Type" : config["fttype"],
    "Dataset" :  config["dataset_name"],
    "Epochs" : trainer.current_epoch+1,
    "Patience" : config["patience"],
    "Learning Rate" : config["learning_rate"],
    "Final Epsilon" : final_epsilon,
    "Average Training Time" : avg_time,
    "Average memory (RSS)/epoch" :avg_mem,
    "Average GPU memory/epoch" : avg_gpu_mem,
    "Training Accuracy" : train_acc,
    "Validation Accuracy" : val_acc,
    "Total Parameters" : train_params,
    "Best model path: ": model_checkpoint_callback.best_model_path,
}
df = pd.DataFrame(list(results.items()), columns=['metric', 'value'])   
print(df)
filename = f"{config['fttype']}_{config['epsilon']}.csv"
df.to_csv(f'./{config["fttype"]}_Results/{filename}', index=True)
