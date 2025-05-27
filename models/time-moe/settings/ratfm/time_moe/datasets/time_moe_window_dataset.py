#!/usr/bin/env python
# -*- coding:utf-8 _*-
import random
import numpy as np

from time_moe.datasets.ts_dataset import TimeSeriesDataset


class TimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0, **kwrags):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length
        self.window_size_plus_one = self.window_size + 1

        num_seqs = len(self.dataset)  # The number of time-seris in the specific domain
        ##print('num_seqs', num_seqs) 
        iterator = range(num_seqs)
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=num_seqs)
        except ImportError:
            pass
        self.sub_seq_indexes = []
        for seq_idx in iterator:  # sequence index
            n_points = self.dataset.get_sequence_length_by_idx(seq_idx)  # The number of data points of time-series(sequence index)
            ##print('n_points', n_points)
            for offset_idx in range(0, n_points, self.window_size_plus_one):
                ##print('offset_idx', offset_idx)
                remaining_points = n_points - offset_idx
                ##print('remaining_points', remaining_points)
                if remaining_points < self.window_size_plus_one:
                    break
                self.sub_seq_indexes.append((seq_idx, offset_idx))

        #print('sub_seq_indexes', len(self.sub_seq_indexes), self.sub_seq_indexes)

    def __len__(self):
        return len(self.sub_seq_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    def __getitem__(self, seq_idx):
        seq_i, offset_i = self.sub_seq_indexes[seq_idx]
        ##print('####################################')
        ##print("seq_i", seq_i, "offset_i", offset_i)
        seq = self.dataset[seq_i][offset_i: offset_i + self.window_size_plus_one]
        #print('seq', seq)
        seq = np.array(seq, dtype=np.float32)

        #loss_mask = np.ones(len(seq) - 1, dtype=np.int32)
        zeros = np.zeros(512 + 96 + 512 - 1, dtype=np.int32) 
        ones = np.ones(96 + 1, dtype=np.int32)
        loss_mask = np.concatenate((zeros, ones))
        #print('seq', seq.shape, 'loss_mask', loss_mask.shape)

        n_pad = self.window_size_plus_one - len(seq)
        #print('n_pad', n_pad)
        if n_pad > 0:
            seq = np.pad(seq, (0, n_pad), 'constant', constant_values=0)
            loss_mask = np.pad(loss_mask, (0, n_pad), 'constant', constant_values=0)
        ##print(seq[:-1].shape, seq[1:].shape, loss_mask.shape)
        #print(seq[:-1], seq[1:], np.sum(loss_mask))

        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
            'loss_masks': loss_mask
        }


class UniversalTimeMoEWindowDataset:
    """
    A dataset that generates windows of time series data with pack technique.
    """
    def __init__(self, dataset: TimeSeriesDataset, context_length: int, prediction_length: int = 0,
                 shuffle: bool = False):
        self.dataset = dataset
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.window_size = context_length + prediction_length

        self.window_info_list = []
        n_seqs = len(self.dataset)

        cur_window_info = []
        num_cur_remaining_points = self.window_size

        iterator = range(n_seqs)
        if shuffle:
            iterator = list(iterator)
            random.shuffle(iterator)

        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=n_seqs)
        except ImportError:
            pass

        for seq_idx in iterator:
            seq_len = self.dataset.get_sequence_length_by_idx(seq_idx)
            remaining_seq_len = seq_len
            while remaining_seq_len > 0:
                if remaining_seq_len < num_cur_remaining_points:
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, remaining_seq_len)
                    )

                    # update states
                    num_cur_remaining_points -= remaining_seq_len
                    remaining_seq_len = 0
                else:
                    # add the part of this seq to cur_window
                    cur_window_info.append(
                        (seq_idx, seq_len - remaining_seq_len, num_cur_remaining_points)
                    )

                    # update states
                    remaining_seq_len -= num_cur_remaining_points
                    self.window_info_list.append(cur_window_info)

                    # reset current window
                    num_cur_remaining_points = self.window_size
                    cur_window_info = []

        if num_cur_remaining_points > 0:
            # drop last batch for speed-up
            pass

    def __len__(self):
        return len(self.window_info_list)

    def __getitem__(self, window_idx):
        window_info = self.window_info_list[window_idx]
        seq = []
        for seq_idx, start_idx_in_seq, offset in window_info:
            part_seq = self.dataset[seq_idx][start_idx_in_seq: start_idx_in_seq + offset]
            seq.append(part_seq)
        if len(seq) == 1:
            seq = seq[0]
            if not isinstance(seq, np.ndarray):
                seq = np.array(seq, dtype=np.float32)
            else:
                seq = seq.astype(np.float32)
        else:
            seq = np.concatenate(seq, axis=0, dtype=np.float32)
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:],
        }
