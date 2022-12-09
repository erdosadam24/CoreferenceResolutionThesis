import os
import torch
import logging
from helperbot import BaseBot
from pathlib import Path
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from gapdataset import GAPDataset
from gapmodel import GAPModel
from utils import collate_examples

os.environ["SEED"] = "33223"
BERT_MODEL = "bert-large-uncased"
UNCASED = True

class GAPBot(BaseBot):
    def __init__(self, model, train_loader, val_loader, *, optimizer, clip_grad=0,
        avg_window=100, log_dir="./cache/logs/", log_level=logging.INFO,
        checkpoint_dir="./cache/model_cache/", batch_idx=0, echo=False,
        device="cuda:0", use_tensorboard=False):
        super().__init__(
            model, train_loader, val_loader, 
            optimizer=optimizer, clip_grad=clip_grad,
            log_dir=log_dir, checkpoint_dir=checkpoint_dir, 
            batch_idx=batch_idx, echo=echo,
            device=device, use_tensorboard=use_tensorboard
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loss_format = "%.6f"
        
    def extract_prediction(self, tensor):
        return tensor
    
    def snapshot(self):
        """Override the snapshot method because Kaggle kernel has limited local disk space."""
        loss = self.eval(self.val_loader)
        loss_str = self.loss_format % loss
        self.logger.info("Snapshot loss %s", loss_str)
        self.logger.tb_scalars(
            "losses", {"val": loss},  self.step)
        target_path = (
            self.checkpoint_dir / "best.pth")        
        if not self.best_performers or (self.best_performers[0][0] > loss):
            torch.save(self.model.state_dict(), target_path)
            self.best_performers = [(loss, target_path, self.step)]
            self.logger.info("Saving checkpoint %s...", target_path)
        assert Path(target_path).exists()
        return loss

def defaultbot():
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=UNCASED,
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
    )
    tokenizer.vocab["[A]"] = -1
    tokenizer.vocab["[B]"] = -1
    tokenizer.vocab["[P]"] = -1
    df_train = pd.read_csv("gap-development.tsv", delimiter="\t")
    df_val = pd.read_csv("gap-validation.tsv", delimiter="\t")
    train_ds = GAPDataset(df_train, tokenizer)
    val_ds = GAPDataset(df_val, tokenizer)
    train_loader = DataLoader(
        train_ds,
        collate_fn = collate_examples,
        batch_size=20,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        collate_fn = collate_examples,
        batch_size=50,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )
    model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bot = GAPBot(
        model, train_loader, val_loader,
        optimizer=optimizer, echo=True,
        avg_window=25
    )
    return bot