import os
import sys
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from common.config import ucr_root_path


def load_and_split_train_dataset(file_path, context_length, scale=True):
    """
    Load the normal (training) data from a given UCR Anomaly file and split it into fixed-length segments.

    Parameters
    ----------
    file_path : str
        Full path to the UCR Anomaly .txt file.
    context_length : int, optional
        The fixed length for each segment. Default is 96.
    scale : bool, optional
        Whether to standardize the data using StandardScaler. Default is True.

    Returns
    -------
    all_segments : list of np.ndarray
        List of segments extracted from the training portion of the time series.
    all_anomalies : list of list
        A list of empty anomaly intervals for each segment (since training data is assumed normal).
    """
    file_name = os.path.basename(file_path)
    fields = file_name.split('_')

    meta_data = {
        'name': '_'.join(fields[:4]),
        'train_end': int(fields[4]),
    }

    # Load time series from file
    with open(file_path) as f:
        lines = f.readlines()
        if len(lines) == 1:
            values = lines[0].strip()
            Y = np.array([eval(y) for y in values.split(" ") if len(y) > 1]).reshape(1, -1)
        else:
            Y = np.array([eval(line.strip()) for line in lines]).reshape(1, -1)

    Y = Y.reshape(-1, 1)
    Y_train = Y[:meta_data['train_end']]

    # Optional scaling
    if scale:
        scaler = StandardScaler()
        scaler.fit(Y_train)
        Y_train = scaler.transform(Y_train)

    # Split into fixed-length segments
    length = Y_train.shape[0]
    num_chunks = length // context_length

    all_segments = []
    all_anomalies = []

    for i in range(num_chunks):
        start = i * context_length
        end = start + context_length
        if end > length:
            break

        segment = Y_train[start:end].copy()
        all_segments.append(segment)
        all_anomalies.append([[]])  # No anomalies

    return all_segments, all_anomalies


def save_dataset_as_pickle(output_dir, series_list, anomaly_list):
    """
    Save the segmented dataset to a pickle file.

    Parameters
    ----------
    output_dir : str
        Directory to save the `data.pkl` file.
    series_list : list of np.ndarray
        Segmented time series data.
    anomaly_list : list of list
        Corresponding anomaly intervals (empty in this case).
    """
    os.makedirs(output_dir, exist_ok=True)
    data = {
        'series': series_list,
        'anom': anomaly_list
    }
    output_path = os.path.join(output_dir, 'data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved: {output_path}")


def main():
    input_dir = ucr_root_path
    output_dir = "../../data/gpt-4o/train/"
    context_length = 96

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"Number of input files: {len(files)}")

    for i, file_name in enumerate(files):
        full_path = os.path.join(input_dir, file_name)

        try:
            # Load and split training data
            series_list, anomaly_list = load_and_split_train_dataset(
                full_path,
                context_length=context_length,

            )

            # Create output folder name based on file prefix
            save_dir_name = '_'.join(file_name.split('_')[:4])
            save_dir = os.path.join(output_dir, save_dir_name)

            save_dataset_as_pickle(save_dir, series_list, anomaly_list)

        except Exception as e:
            print(f"Error in {file_name}: {e}")


if __name__ == "__main__":
    main()

