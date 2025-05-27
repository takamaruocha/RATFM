from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import random
import os
from openai_api import send_openai_request
from sklearn.metrics import precision_score, recall_score, f1_score
import sys
sys.path.append('affiliation-metrics-py')
from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events


def parse_output(output: str) -> dict:
    """Parse the output of the AD model.

    Args:
        output: The output of the AD model.

    Returns:
        A dictionary containing the parsed output.
    """
    import json
    
    # Trim the output string
    trimmed_output = output[output.index('['):output.rindex(']') + 1]
    # Try to parse the output as JSON
    parsed_output = json.loads(trimmed_output)
    
    # Validate the output: list of dict with keys start and end
    for item in parsed_output:
        if not isinstance(item, dict):
            raise ValueError("Parsed output contains non-dict items")
        if 'start' not in item or 'end' not in item:
            raise ValueError("Parsed output dictionaries must contain 'start' and 'end' keys")
    
    return parsed_output


def interval_to_vector(interval, start=0, end=1000):
    anomalies = np.zeros((end - start, 1))
    for entry in interval:
        if type(entry) is not dict:
            assert len(entry) == 2
            entry = {'start': entry[0], 'end': entry[1]}
        entry['start'] = int(entry['start'])
        entry['end'] = int(entry['end'])
        entry['start'] = np.clip(entry['start'], start, end)
        entry['end'] = np.clip(entry['end'], entry['start'], end)
        anomalies[entry['start']:entry['end']] = 1
    return anomalies


def vector_to_interval(vector):
    intervals = []
    in_interval = False
    start = 0
    for i, value in enumerate(vector):
        if value == 1 and not in_interval:
            start = i
            in_interval = True
        elif value == 0 and in_interval:
            intervals.append((start, i))
            in_interval = False
    if in_interval:
        intervals.append((start, len(vector)))
    return intervals


def create_color_generator(exclude_color='blue'):
    # Get the default color list
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
    # Filter out the excluded color
    filtered_colors = [color for color in default_colors if color != exclude_color]
    # Create a generator that yields colors in order
    return (color for color in filtered_colors)


def plot_series_and_predictions(
    series: np.ndarray,
    gt_anomaly_intervals: list[list[tuple[int, int]]],
    global_min: float,
    global_max: float,
    anomalies: Optional[dict] = None,
    single_series_figsize: tuple[int, int] = (20, 3),
    gt_ylim: tuple[int, int] = (-1, 1),
    gt_color: str = 'steelblue',
    anomalies_alpha: float = 0.5
) -> None:
    plt.figure(figsize=single_series_figsize)
    
    color_generator = create_color_generator()
    
    def get_next_color(color_generator):
        try:
            # Return the next color
            return next(color_generator)
        except StopIteration:
            # If all colors are used, reinitialize the generator and start over
            color_generator = create_color_generator()
            return next(color_generator)

    num_anomaly_methods = len(anomalies) if anomalies else 0
    ymin_max = [
        (
            i / num_anomaly_methods * 0.5 + 0.25,
            (i + 1) / num_anomaly_methods * 0.5 + 0.25,
        )
        for i in range(num_anomaly_methods)
    ]
    ymin_max = ymin_max[::-1]

    for i in range(series.shape[1]):
        # plt.ylim(gt_ylim)
        ymin, ymax = global_min, global_max
        # ymin, ymax = np.min(series[:, i]), np.max(series[:, i])
        padding = 0.1 * (ymax - ymin)  # 少しだけ余白
        plt.ylim(ymin - padding, ymax + padding)
        plt.plot(series[:, i], color=gt_color)

        if gt_anomaly_intervals is not None:
            for start, end in gt_anomaly_intervals[i]:
                plt.axvspan(start, end, alpha=0.2, color=gt_color)

        if anomalies is not None:
            for idx, (method, anomaly_values) in enumerate(anomalies.items()):
                if anomaly_values.shape == series.shape:
                    anomaly_values = np.nonzero(anomaly_values[:, i])[0].flatten()
                ymin, ymax = ymin_max[idx]
                random_color = get_next_color(color_generator)  # Use the function to get a random color
                for anomaly in anomaly_values:
                    plt.axvspan(anomaly, anomaly + 1, ymin=ymin, ymax=ymax, alpha=anomalies_alpha, color=random_color, lw=0)
                plt.plot([], [], color=random_color, label=method)

    plt.tight_layout()
    if anomalies is not None:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return plt.gcf()


def generate_batch_AD_requests(
    model_name: str,
    data_name: str,
    request_func: callable,
    variant: str = "standard",
    entity = "160_UCR_Anomaly_TkeepThirdMARS",
):
    import json
    import time
    import pickle
    import os
    from loguru import logger
    from data.synthetic import SyntheticDataset
    from tqdm import trange
    
    results_dir = f'results/{data_name}_batch/{entity}/{model_name}/'
    base_dir = "/Storage2/maru/datasets/UCR_Anomaly_Archive/AnomLLM/"
    data_dir = os.path.join(base_dir, "eval", entity)
    train_dir = os.path.join(base_dir, "train", entity)
    jsonl_fn = os.path.join(results_dir, variant + '_requests.jsonl')
    os.makedirs(results_dir, exist_ok=True)
    
    # Remove the existing jsonl file
    if os.path.exists(jsonl_fn):
        os.remove(jsonl_fn)
    
    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()

    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()

    # 全部を縦方向（0軸）に結合する
    all_series = np.vstack(eval_dataset.series)

    # 最小値と最大値を取得
    global_min = np.min(all_series)
    global_max = np.max(all_series)

    print(global_min, global_max)

    for i in trange(1, len(eval_dataset) + 1):
        idx = f"{str(i).zfill(5)}"
        body = request_func(
            eval_dataset.series[i - 1],
            train_dataset,
            entity,
            global_min,
            global_max
        )
        body['model'] = model_name
        custom_id = f"{data_name}_{model_name}_{variant}_{idx}"
        request = {
            "custom_id": custom_id,
            "body": body,
            "method": "POST",
            "url": "/v1/chat/completions",
        }
        # Write the result to jsonl
        with open(jsonl_fn, 'a') as f:
            json.dump(request, f)
            f.write('\n')
    logger.info(f"Succesfully generated {len(eval_dataset)} AD requests and saved them to {jsonl_fn}.")
    return jsonl_fn


def view_base64_image(base64_string):
    import base64
    from io import BytesIO
    from PIL import Image
    import matplotlib.pyplot as plt

    # Decode the base64 string to binary data
    image_data = base64.b64decode(base64_string)
    
    # Convert binary data to an image
    image = Image.open(BytesIO(image_data))
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.show()


def display_messages(messages):
    from IPython.display import display, HTML
    
    html_content = "<div style='font-family: Arial, sans-serif;'>"

    for message in messages:
        role = message['role'].upper()
        html_content += f"<p><strong>{role}:</strong></p>"
        if isinstance(message['content'], str):
            message['content'] = [{'type': 'text', 'text': message['content']}]
        for content in message['content']:
            if content['type'] == 'text':
                text = content['text']
                html_content += f"<p style='white-space: pre-wrap;'>{text}</p>"
            elif content['type'] == 'image_url':
                image_url = content['image_url']['url']
                html_content += (
                    f"<div style='text-align: center;'><img src='{image_url}' alt='User Image' "
                    "style='margin: 10px auto; display: block; max-width: 50%;'/></div>"
                )

    html_content += "</div>"

    display(HTML(html_content))


def highlight_by_ranking(df):
    def generate_html_color(value, min_val, midpoint, max_val):
        """ Helper function to generate HTML color based on relative ranking. """
        # Normalize value to get a color gradient
        if value <= midpoint:
            ratio = (value - min_val) / (midpoint - min_val)
            if np.isnan(ratio):
                ratio = 0
            r = int(0 + 127 * ratio)
            g = int(255 - 127 * ratio)
            b = 0
        else:
            ratio = (value - midpoint) / (max_val - midpoint)
            if np.isnan(ratio):
                ratio = 0
            r = int(127 + 128 * ratio)
            g = int(127 - 127 * ratio)
            b = 0
        return f'rgb({r},{g},{b})'

    # Convert to DataFrame if it's a Series (single column)
    if isinstance(df, pd.Series):
        df = df.to_frame()

    styled_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        # Rank the values in the column, larger number ranks lower (is worse)
        rankings = df[col].rank(method='min', ascending=False)
        
        min_rank, max_rank = rankings.min(), rankings.max()
        mid_rank = (max_rank + min_rank) / 2
        
        styled_col = [
            f'<span style="color:{generate_html_color(rank, min_rank, mid_rank, max_rank)};">{value * 100:.2f}</span>'
            for value, rank in zip(df[col], rankings)
        ]
        styled_df[col] = styled_col

    # If input was a Series, return a Series
    if len(df.columns) == 1 and isinstance(df, pd.DataFrame):
        return styled_df[df.columns[0]]
    
    # Replace precision in column names by prec
    styled_df.columns = [col.replace('precision', 'PRE').replace('recall', 'REC').replace('f1', 'F1') for col in styled_df.columns]
    
    return styled_df


def process_dataframe(df):
    import re
    
    # Function to extract model name and variant
    def extract_model_variant(text):
        match = re.match(r'(.*?)\s*\((.*?)\)', text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return text, ''

    # Split the index into model and variant
    df.reset_index(inplace=True)
    df['model'], df['variant'] = zip(*df['index'].apply(extract_model_variant))
    
    # Drop the original index column
    df.drop('index', axis=1, inplace=True)
    
    # Sort by model and variant
    df = df.sort_values(['model', 'variant'])
    
    # Set model and variant as index
    df.set_index(['model', 'variant'], inplace=True)
    
    return df


def compute_metrics(gt, prediction):
    # Check if both gt and prediction are empty
    if prediction is None:
        print("A")
        metrics = {
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'affi precision': 0,
            'affi recall': 0,
            'affi f1': 0
        }
    
    # elif np.count_nonzero(gt) == 0 and np.count_nonzero(prediction) == 0:
    #  gtとpredictionが全て0（正常）の場合は、異常検知に成功（全ての評価指標が1）
    #     print("B")
    #     metrics = {
    #         'precision': 1,
    #         'recall': 1,
    #         'f1': 1,
    #         'affi precision': 1,
    #         'affi recall': 1,
    #         'affi f1': 1
    #     }
    # Check if only gt is empty
    # gtまたはpredictionに異常が一つもない→recallまたはprecisionの分母が0になる
    # elif np.count_nonzero(gt) == 0 or np.count_nonzero(prediction) == 0:
    #     print("C")
    #     metrics = {
    #         'precision': 0,
    #         'recall': 0,
    #         'f1': 0,
    #         'affi precision': 0,
    #         'affi recall': 0,
    #         'affi f1': 0
    #     }
    
    else:
        print("D")
        precision = precision_score(gt, prediction, zero_division=0)
        recall = recall_score(gt, prediction, zero_division=0)
        f1 = f1_score(gt, prediction, zero_division=0)
        
        # events_pred = convert_vector_to_events(prediction)
        # events_gt = convert_vector_to_events(gt)
        # Trange = (0, len(prediction))
        # aff = pr_from_events(events_pred, events_gt, Trange)
        
        # Calculate affiliation F1
        # if aff['precision'] + aff['recall'] == 0:
        #     affi_f1 = 0
        # else:
        #     affi_f1 = 2 * (aff['precision'] * aff['recall']) / (aff['precision'] + aff['recall'])
        
        metrics = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            # 'affi precision': round(aff['precision'], 3),
            # 'affi recall': round(aff['recall'], 3),
            # 'affi f1': round(affi_f1, 3)
        }
    return metrics


def styled_df_to_latex(styled_df, caption, label):
    def extract_color(html):
        import re
        color_match = re.search(r'color:rgb\((\d+),(\d+),(\d+)\);', html)
        if color_match:
            return tuple(map(int, color_match.groups()))
        return (0, 0, 0)  # Default to black if no color is found

    def rgb_to_latex_color(rgb):
        return f"\\color[RGB]{{{rgb[0]},{rgb[1]},{rgb[2]}}}"

    def format_number(num):
        return f"\\small{{{num:.2f}}}"  # Apply \tiny to the number

    def format_header(headers):
        top_row = []
        bottom_row = []
        for header in headers:
            parts = header.split()
            if len(parts) > 1:
                top_row.append(f"\\small\\fontfamily{{cmtt}}\\selectfont{{{parts[0]}}}")
                bottom_row.append(f"\\small\\fontfamily{{cmtt}}\\selectfont{{{' '.join(parts[1:])}}}")
            else:
                top_row.append(f"\\small\\fontfamily{{cmtt}}\\selectfont{{{header}}}")
                bottom_row.append('')
        return ' & '.join(top_row) + ' \\\\', ' & '.join(bottom_row) + ' \\\\'

    def format_index(idx):
        def replace(s):
            return s.replace('classical ', '')
        if isinstance(idx, tuple):
            return replace(' '.join(idx))
        return replace(idx)

    def camel_style_with_dash(s):
        def format_word(word):
            if len(word) == 3:
                return word.upper()
            return word.capitalize()

        words = s.split('-')
        return '-'.join(format_word(word) for word in words)

    latex_lines = [
        "\\begin{longtable}{" + "l" * (styled_df.index.nlevels) + "r" * (len(styled_df.columns)) + "}",
        "\\caption{" + caption + "} \\label{tab:" + label + "} \\\\",
        "\\toprule"
    ]

    top_header, bottom_header = format_header(styled_df.columns)
    latex_lines.extend([
        "&" * styled_df.index.nlevels + " " + top_header,
        "&" * styled_df.index.nlevels + " " + bottom_header + " \\endfirsthead",
        "\\multicolumn{" + str(styled_df.index.nlevels + len(styled_df.columns)) + "}{c}{\\tablename\\ \\thetable\\ -- continued from previous page} \\\\", # noqa
        "\\toprule",
        "&" * styled_df.index.nlevels + " " + top_header,
        "&" * styled_df.index.nlevels + " " + bottom_header + " \\endhead",
        "\\midrule \\multicolumn{" + str(styled_df.index.nlevels + len(styled_df.columns)) + "}{r}{Continued on next page} \\\\ \\endfoot",
        "\\bottomrule \\endlastfoot",
        "\\midrule"
    ])

    prev_model = None
    model_row_count = 0
    for i, (idx, row) in enumerate(styled_df.iterrows()):
        cell_color = "\\cellcolor{gray!15}" if i % 2 == 0 else ""
        row_values = []
        for value in row:
            color = extract_color(value)
            numeric_value = float(value.split('>')[1].split('<')[0])
            latex_color = rgb_to_latex_color(color)
            formatted_value = format_number(numeric_value)
            row_values.append(f"{cell_color}{latex_color}{formatted_value}")

        if isinstance(idx, tuple):
            model, variant = idx
            if model != prev_model:
                if prev_model is not None:
                    latex_lines.append("\\midrule")
                model_row_count = 1
                latex_lines.append(f"\\multirow{{-1}}{{*}}{{\\footnotesize\\fontfamily{{cmtt}}\\selectfont{{{camel_style_with_dash(format_index(model))}}}}} & {cell_color}\\footnotesize\\fontfamily{{cmtt}}\\selectfont{{{camel_style_with_dash(format_index(variant))}}} & " + " & ".join(row_values) + " \\\\")  # noqa
                prev_model = model
            else:
                model_row_count += 1
                latex_lines.append(
                    f"& {cell_color}\\footnotesize\\fontfamily{{cmtt}}\\selectfont{{{camel_style_with_dash(format_index(variant))}}} & "
                    + " & ".join(row_values)
                    + " \\\\"
                )
        else:
            latex_lines.append(
                f"\\footnotesize\\fontfamily{{cmtt}}\\selectfont{{{camel_style_with_dash(format_index(idx))}}} & {cell_color}"
                + " & ".join(row_values)
                + " \\\\"
            )

    latex_lines.append("\\end{longtable}")

    return "\n".join(latex_lines)


def load_results(result_fn, raw=False, postprocess_func: callable = None):
    """
    Load and process results from a result JSON lines file.

    Parameters
    ----------
    result_fn : str
        The filename of the JSON lines file containing the results.
    raw : bool, optional
        If True, return raw JSON objects. If False, parse the response
        and convert it to a vector. Default is False.
    postprocess_func : callable, optional
        A function to postprocess the results (e.g., scaling down). Default is None.

    Returns
    -------
    list
        A list of processed results. Each item is either a raw JSON object
        or a vector representation of anomalies, depending on the
        `raw` parameter.

    Notes
    -----
    The function attempts to parse each line in the file. If parsing fails,
    it appends an empty vector to the results.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    JSONDecodeError
        If a line in the file is not valid JSON.
    """
    import json
    import pandas as pd
    from utils import parse_output, interval_to_vector
    
    if postprocess_func is None:
        postprocess_func = lambda x: x
    
    with open(result_fn, 'r') as f:
        results = []
        for line in f:
            info = json.loads(line)
            if raw:
                results.append(info)
            else:
                try:
                    response_parsed = parse_output(postprocess_func(info['response']))
                    results.append(interval_to_vector(response_parsed, start=0, end=96))
                except Exception:
                    results.append(None)
                    continue
            
    return results


def collect_results(directory, raw=False, ignore=[]):
    """
    Collect and process results from JSON lines files in a directory.

    Parameters
    ----------
    directory : str
        The path to the directory containing the JSON lines files.
    raw : bool, optional
        If True, return raw JSON objects. If False, parse the responses.
        Default is False.
    ignore: list[str], optional
        Skip folders containing these names. Default is an empty list.

    Returns
    -------
    dict
        A dictionary where keys are model names with variants, and values
        are lists of processed results from each file.

    Notes
    -----
    This function walks through the given directory, processing each
    `.jsonl` file except those with 'requests' in the filename. It uses
    the directory name as the model name and the filename (sans extension)
    as the variant.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.
    """
    import os
    from config import postprocess_configs

    results = {}
    config = postprocess_configs()
    for root, _, files in os.walk(directory):
        for file in files:
            skip = False
            for ignore_folder in ignore:
                if ignore_folder in root:
                    skip = True
                    break
            if skip:
                continue
            if 'requests' not in file and file.endswith('.jsonl'):
                model_name = os.path.basename(root)
                variant = file.replace('.jsonl', '')
                if variant in config:
                    pf = config[variant]
                else:
                    pf = None
                result_fn = os.path.join(root, file)
                model_key = f'{model_name} ({variant})'
                results[model_key] = load_results(result_fn, raw=raw, postprocess_func=pf)
    return results


def EDA(eval_dataset):
    total_anom = 0
    total = 0
    time_series_without_anomalies = 0
    anomaly_counts = []
    anomaly_lengths = []

    for i in range(400):
        data = eval_dataset[i]
        series_anom = 0
        series_anomaly_count = 0
        
        for start, end in data[0][0]:
            length = end - start
            series_anom += length
            series_anomaly_count += 1
            anomaly_lengths.append(length)
        
        total_anom += series_anom
        total += 1000
        
        if series_anom == 0:
            time_series_without_anomalies += 1
        
        anomaly_counts.append(series_anomaly_count)

    # Calculate statistics
    avg_anomaly_ratio = total_anom / total
    percent_without_anomalies = (time_series_without_anomalies / 400) * 100
    avg_anomalies_per_series = sum(anomaly_counts) / 400
    max_anomalies_in_series = max(anomaly_counts)
    avg_anomaly_length = sum(anomaly_lengths) / len(anomaly_lengths)
    max_anomaly_length = max(anomaly_lengths)

    print(f"Average anomaly ratio: {avg_anomaly_ratio:.4f}")
    print(f"Number of time series without anomalies: {time_series_without_anomalies}")
    print(f"Percentage of time series without anomalies: {percent_without_anomalies:.2f}%")
    print(f"Average number of anomalies per time series: {avg_anomalies_per_series:.2f}")
    print(f"Maximum number of anomalies in a single time series: {max_anomalies_in_series}")
    print(f"Average length of an anomaly: {avg_anomaly_length:.2f}")
    print(f"Maximum length of an anomaly: {max_anomaly_length}")


if __name__ == '__main__':
    from data.synthetic import SyntheticDataset
    
    for name in ['point', 'range', 'trend', 'freq', 'noisy-point', 'noisy-trend', 'noisy-freq']:
        print(f"Dataset: {name}")
        eval_dataset = SyntheticDataset(f'data/synthetic/{name}/eval/')
        eval_dataset.load()
        EDA(eval_dataset)
