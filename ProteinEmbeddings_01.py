import re
import os

import torch
import pandas as pd
from transformers import BertModel, BertTokenizer


def clean_sequence(seq):
    """
    Clean a raw amino acid sequence string for ProtBert-BFD:
    - Uppercase
    - Replace rare/ambiguous amino acids U, Z, O, B with X
    - Insert spaces between residues (ProtTrans convention)
    """
    seq = str(seq).strip().upper()
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = " ".join(list(seq))
    return seq


def mean_pool(last_hidden_state, attention_mask):
    """
    Mean pooling over the sequence length dimension, masking out padding
    AND special tokens ([CLS], [SEP]) at positions 0 and -1.
    last_hidden_state: (batch, seq_len, hidden_dim)
    attention_mask:    (batch, seq_len)
    returns:           (batch, hidden_dim)
    """
    # Zero out the [CLS] (pos 0) and [SEP] (last non-padding pos) tokens
    special_token_mask = attention_mask.clone()
    special_token_mask[:, 0] = 0  # mask [CLS]
    # Find last real token per sequence and mask it ([SEP])
    seq_lengths = attention_mask.sum(dim=1) - 1  # index of [SEP]
    for i, sep_idx in enumerate(seq_lengths):
        special_token_mask[i, sep_idx] = 0

    input_mask_expanded = special_token_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1)
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


def compute_protein_embeddings(
    input_csv_path,
    output_csv_path,
    batch_size=8,
    device=None,
):
    """
    Read a CSV with:
    - column 0: something (ignored here)
    - column 1: something (ignored here)
    - column 2: amino acid sequence
    - column 3: affinity (target label)

    Use ProtBert-BFD to compute mean-pooled embeddings for each sequence
    and save them together with the affinity to output_csv_path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df = pd.read_csv(input_csv_path)
    if df.shape[1] < 4:
        raise ValueError(
            "Expected at least 4 columns, but got {} in {}".format(df.shape[1], input_csv_path)
        )

    # Extract sequences and affinities by column index (3rd and 4th columns)
    sequences = df.iloc[:, 2].tolist()
    affinities = df.iloc[:, 3].tolist()

    # Clean sequences for ProtTrans tokenization
    cleaned_sequences = [clean_sequence(s) for s in sequences]

    # Load ProtBert-BFD tokenizer & model
    model_name = "Rostlab/prot_bert_bfd"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    all_embeddings = []
    n = len(cleaned_sequences)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_seqs = cleaned_sequences[start:end]

            encoded = tokenizer(
                batch_seqs,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            batch_pooled = mean_pool(token_embeddings, attention_mask)  # (batch, hidden_dim)
            all_embeddings.append(batch_pooled.cpu())

    # Concatenate all batches: (N, hidden_dim)
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    hidden_dim = embeddings_tensor.size(1)

    # Build output DataFrame
    row_ids = df.index.to_list()
    out_dict = {
        "row_id": row_ids,
        "sequence": sequences,
        "affinity": affinities,
    }

    # Add embedding columns
    for i in range(hidden_dim):
        out_dict["emb_{}".format(i)] = embeddings_tensor[:, i].numpy()

    out_df = pd.DataFrame(out_dict)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv_path)
    if out_dir == "":
        out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(output_csv_path, index=False)
    print("Done! Saved embeddings to:", output_csv_path)
    return out_df


# FIX 2: Guard so this only runs when executed directly, not on import
if __name__ == "__main__":
    compute_protein_embeddings(
        input_csv_path="small_data.csv",
        output_csv_path="data/protein_embeddings_small.csv",
        batch_size=16,
        device="cpu",
    )