import os
import pandas as pd
import torch
from helperbot import TriangularLR
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader

from gapdataset import GAPDataset
from gapmodel import GAPModel
from gapbot import GAPBot

from utils import collate_examples, set_trainable, convert_to_results, calculate_score, calculate_accuracy

os.environ["SEED"] = "33223"
BERT_MODEL = "bert-large-uncased"
UNCASED = True

if __name__ == '__main__':

    df_train = pd.read_csv("merged_10k.tsv", delimiter="\t")
    df_val = pd.read_csv("gap-validation.tsv", delimiter="\t")
    df_test = pd.read_csv("gap-development.tsv", delimiter="\t")
    sample_sub = pd.read_csv("./input/sample_submission_stage_1.csv")
    assert sample_sub.shape[0] == df_test.shape[0]

    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=UNCASED,
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
    )
    # These tokens are not actually used, so we can assign arbitrary values.
    tokenizer.vocab["[A]"] = -1
    tokenizer.vocab["[B]"] = -1
    tokenizer.vocab["[P]"] = -1

    train_ds = GAPDataset(df_train, tokenizer)
    val_ds = GAPDataset(df_val, tokenizer)
    test_ds = GAPDataset(df_test, tokenizer)
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
    test_loader = DataLoader(
        test_ds,
        collate_fn = collate_examples,
        batch_size=50,
        num_workers=2,
        pin_memory=True,
        shuffle=False
    )

    model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
    # You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)
    # set_trainable(model.bert.encoder.layer[23], True)
    set_trainable(model.bert, False)
    set_trainable(model.head, True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bot = GAPBot(
        model, train_loader, val_loader,
        optimizer=optimizer, echo=True,
        avg_window=25
    )

    #bot.load_model("./cache/model_cache/best.pth")

    steps_per_epoch = len(train_loader) 
    n_steps = steps_per_epoch * 20
    bot.train(
        n_steps,
        log_interval=steps_per_epoch // 20,
        snapshot_interval=steps_per_epoch,
        scheduler=TriangularLR(
            optimizer, 20, ratio=2, steps_per_cycle=n_steps)
    )

    # Load the best checkpoint
    bot.load_model(bot.best_performers[0][1])

    # Evaluate on the test dataset
    print(bot.eval(test_loader))

    # Extract predictions to the test dataset
    preds = bot.predict(test_loader)

    # Create submission file
    df_sub = pd.DataFrame(torch.softmax(preds, -1).cpu().numpy().clip(1e-3, 1-1e-3), columns=["A", "B", "NEITHER"])
    df_sub["ID"] = df_test.ID
    df_sub.to_csv("submission.csv", index=False)
    convert_to_results('submission.csv').to_csv('prediction.csv', index=False)

    print('Kaggle score: ' + str(calculate_score('right_answers.csv','submission.csv')))
    print('Accuracy: ' + str(calculate_accuracy('right_answers.csv','prediction.csv')))