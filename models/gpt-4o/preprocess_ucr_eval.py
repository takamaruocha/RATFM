import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.config import ucr_root_path


def load_and_split_dataset(file_path, scale=True, context_length=96, forecast_horizon=96, initial_seq_start=1120):
    """
    Load a time series file and split the test portion into overlapping windows,
    recording anomaly intervals for each segment.

    Parameters
    ----------
    file_path : str
        Path to the .txt file.
    scale : bool
        Whether to normalize using the training portion.
    context_length : int
        Total sequence length for each window.
    forecast_horizon : int
        Number of forecast steps.
    initial_seq_start : int
        Start index of the first window in the test data.

    Returns
    -------
    list of np.ndarray
        List of input sequences.
    list of list of tuples
        List of anomaly interval annotations per window.
    """
    file_name = os.path.basename(file_path)
    fields = file_name.split('_')
    meta_data = {
        'name': '_'.join(fields[:4]),
        'train_end': int(fields[4]),
        'anomaly_start_in_test': int(fields[5]) - int(fields[4]),
        'anomaly_end_in_test': int(fields[6][:-4]) - int(fields[4]),
    }

    # Load data
    with open(file_path) as f:
        Y = f.readlines()
        if len(Y) == 1:
            Y = Y[0].strip()
            Y = np.array([eval(y) for y in Y.split(" ") if len(y) > 1]).reshape((1, -1))
        elif len(Y) > 1:
            Y = np.array([eval(y.strip()) for y in Y]).reshape((1, -1))

    Y = Y.reshape(-1, 1)
    Y_train = Y[:meta_data['train_end']]

    if scale:
        scaler = StandardScaler()
        scaler.fit(Y_train)
        Y = scaler.transform(Y)

    Y_test = Y[meta_data['train_end']:]
    length = Y_test.shape[0]

    if initial_seq_start >= length:
        raise ValueError(f"initial_seq_start={initial_seq_start} exceeds test data length {length}.")

    # Create binary anomaly label for test data
    label = np.zeros(length)
    label[meta_data['anomaly_start_in_test']:meta_data['anomaly_end_in_test'] + 1] = 1

    num_chunks = (length - initial_seq_start) // forecast_horizon + 1
    all_segments = []
    all_anomalies = []

    for i in range(num_chunks):
        seq_start = (initial_seq_start - (context_length - forecast_horizon)) + forecast_horizon * i
        seq_end = seq_start + context_length

        if seq_end > length:
            seq_end = length
            seq_start = seq_end - context_length

        segment = Y_test[seq_start:seq_end].copy()
        segment_label = label[seq_end - forecast_horizon:seq_end]

        # Extract anomaly intervals
        anomaly_intervals = []
        in_anomaly = False
        for j in range(forecast_horizon):
            if segment_label[j] == 1 and not in_anomaly:
                start = j
                in_anomaly = True
            elif segment_label[j] == 0 and in_anomaly:
                anomaly_intervals.append((start, j))
                in_anomaly = False
        if in_anomaly:
            anomaly_intervals.append((start, forecast_horizon))

        all_segments.append(segment)
        all_anomalies.append([anomaly_intervals])  # single univariate sensor

    return all_segments, all_anomalies


def save_dataset_as_pickle(output_dir, series_list, anomaly_list):
    """
    Save the segmented data and anomaly annotations to a pickle file.

    Parameters
    ----------
    output_dir : str
        Directory to save data.pkl in.
    series_list : list
        Segmented input series.
    anomaly_list : list
        Corresponding anomaly interval lists.
    """
    os.makedirs(output_dir, exist_ok=True)
    data_dict = {
        'series': series_list,
        'anom': anomaly_list
    }
    output_path = os.path.join(output_dir, 'data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Saved: {output_path}")


def main():
    input_dir = ucr_root_path
    output_dir = "../../data/gpt-4o/eval/"
    context_length = 96
    forecast_horizon = 96
    initial_seq_start = 1120

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"📂 Number of files to process: {len(files)}")

    for file_name in files:
        full_path = os.path.join(input_dir, file_name)

        try:
            series_list, anomaly_list = load_and_split_dataset(
                full_path,
                scale=True,
                context_length=context_length,
                forecast_horizon=forecast_horizon,
                initial_seq_start=initial_seq_start
            )

            save_dir_name = '_'.join(file_name.split('_')[:4])
            save_dir = os.path.join(output_dir, save_dir_name)

            save_dataset_as_pickle(save_dir, series_list, anomaly_list)

        except Exception as e:
            print(f"❌ Error in {file_name}: {e}")


if __name__ == "__main__":
    main()

