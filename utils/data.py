import os
import pandas as pd

def process(label_path, seq_path, out_label_path, out_seq_path, length_limit=256):
# Rules for processing RNA sequences:
# 1. Remove sequences longer than 256 bases
# 2. Remove sequences with any missing coordinate values
# 3. Remove sequences containing 'X' or '-'
    labels = pd.read_csv(label_path)
    sequences = pd.read_csv(seq_path, engine='python')  # use python engine to handle irregular CSV

    # extract common sequence ID
    labels['seq_id'] = labels['ID'].str.split('_').str[:-1].str.join('_')
    sequences['seq_id'] = sequences['target_id']

    # compute lengths and mark missing coords
    sequences['length'] = sequences['sequence'].str.len()
    labels['missing_flag'] = labels[['x_1','y_1','z_1']].isnull().any(axis=1)

    # mark sequences containing '-' or 'X'
    sequences['invalid_flag'] = sequences['sequence'].str.contains(r'[-X]')

    # aggregate missing info per sequence
    missing = labels.groupby('seq_id', sort=False)['missing_flag']                     .any().reset_index(name='has_missing')

    # merge stats and decide drops
    stats = sequences[['seq_id','length','invalid_flag']].merge(missing, on='seq_id', how='left')
    stats['has_missing'] = stats['has_missing'].fillna(False)
    drop_ids = stats.loc[
        (stats['length'] > length_limit) |
        (stats['has_missing']) |
        (stats['invalid_flag']),
        'seq_id'
    ]

    # filter out unwanted entries
    clean_labels = labels.loc[~labels['seq_id'].isin(drop_ids)]                        .drop(columns=['seq_id','missing_flag'])
    clean_seqs   = sequences.loc[~sequences['seq_id'].isin(drop_ids)]                          .drop(columns=['seq_id','length','invalid_flag'])

    # save cleaned files
    clean_labels.to_csv(out_label_path, index=False)
    clean_seqs.to_csv(out_seq_path, index=False)

if __name__ == '__main__':
    process(
        "../data/raw/train_labels.csv",
        "../data/raw/train_sequences.csv",
        "../data/train_labels.csv",
        "../data/train_sequences.csv",
    )
    process(
        "../data/raw/validation_labels.csv",
        "../data/raw/validation_sequences.csv",
        "../data/valid_labels.csv",
        "../data/valid_sequences.csv",
    )