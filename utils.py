import torch
import re
import numpy as np
import torch.nn as nn
import pandas as pd
from collections import namedtuple

WordWithOffsets = namedtuple("WordWithOffsets", ["word", "offsets"])
WordWithOffset = namedtuple("WordWithOffset", ["word", "offset"])
WordWithOffsetAndScore = namedtuple("WordWithOffsetAndScore", ["word_with_offset", "score"])
Coreference = namedtuple("Coreference", ["id", "pronoun_with_offset", "noun_with_offset", "score"])

def equal_words_with_offset(word1, word2):
    return word1.word == word2.word and word1.offset == word2.offset

def collate_examples(batch, truncate_len=500):
    """Batch preparation.
    
    1. Pad the sequences
    2. Transform the target.
    """
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) + 1 # Account for the [CLS] token
    # Labels
    if len(transposed) == 2:
        return token_tensor, offsets, None
    one_hot_labels = torch.stack([
        torch.from_numpy(x.astype("uint8")) for x in transposed[2]
    ], dim=0)
    _, labels = one_hot_labels.max(dim=1)
    return token_tensor, offsets, labels

def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b

def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))

def process_row(row):
    maxvalue = max(row)
    row = [1 if p == maxvalue else 0 for p in row]
    """
    row = [1 if p == maxvalue and p > 0.5 else 0 for p in row]
    if 1 not in row:
        row[2] = 1
    """
    return row

def convert_to_results(submission_path):
    sam_submission = pd.read_csv(submission_path)
    submission = sam_submission.values[:,:-1].astype(float)

    submission = np.apply_along_axis(func1d=process_row, axis=1, arr=submission)

    df = pd.DataFrame(submission, columns=["A", "B", "NEITHER"])
    df['ID'] = sam_submission.values[:,-1:]
    return df

def preprocess_row(row):
    row = [max(min(p, 1 - 10e-15), 10e-15) for p in row]
    row = [x/sum(row) for x in row]
    return row

def calculate_score(answers_path, submission_path):
    right_answers = pd.read_csv(answers_path)
    sam_submission = pd.read_csv(submission_path)

    y = right_answers.values[:,1:].astype(int)
    submission = sam_submission.values[:,:-1].astype(float)

    submission = np.apply_along_axis(func1d=preprocess_row, axis=1, arr=submission)

    submission = np.log(submission)
    temp = np.multiply(submission, y)

    return np.sum(temp)/-submission.shape[0]

def convert_correct(row):
    row = 1 if 2 in row else 0
    return row

def calculate_accuracy(answers_path, prediction_path):
    right_answers = pd.read_csv(answers_path)
    sam_prediction = pd.read_csv(prediction_path)

    answers = right_answers.values[:,1:].astype(int)

    prediction = sam_prediction.values[:,:-1].astype(int)

    together = answers + prediction

    correct = np.apply_along_axis(func1d=convert_correct, axis=1, arr=together)

    return np.sum(correct)/correct.shape[0]

def calculate_offsets(words, full_text):
    all_together = []
    for word in words:
            all_together.append(WordWithOffsets(word, [m.start() for m in re.finditer(f"\\b{re.escape(word)}\\b", full_text)]))
    return all_together