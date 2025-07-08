# RATFM: Retrieval-augmented Time Series Foundation Model for Anomaly Detection

<img src="images/logo.png" alt="Logo" width="300"/>

## 📖 Introduction
Inspired by the success of large language models (LLMs) in natural language processing, recent research has explored the building of time series foundation models and applied them to tasks such as forecasting, classification, and anomaly detection. However, their performances vary between different domains and tasks. In LLM-based approaches, test-time adaptation using example-based prompting has become common, owing to the high cost of retraining. In the context of anomaly detection, which is the focus of this study, providing normal examples from the target domain can also be effective. However, time series foundation models do not naturally acquire the ability to interpret or utilize examples or instructions, because the nature of time series data used during training does not encourage such capabilities. To address this limitation, we propose a retrieval augmented time series foundation model (RATFM), which enables pretrained time series foundation models to incorporate examples of test-time adaptation. We show that RATFM achieves a performance comparable to that of in-domain fine-tuning while avoiding domain-dependent fine-tuning. Experiments on the UCR Anomaly Archive, a multi-domain dataset including nine domains, confirms the effectiveness of the proposed approach.

## 🧠 Models
<table>
  <thead>
    <tr>
      <th> Base Models </th> <th>Description</th>
    </tr>
  </thead>
  <tr>
    <td> Time-MoE </td> <td> We employed Time-MoE (large), which is a decoder-only time series foundation model with a mixture-of-experts architecture for time series forecasting. </td>
  </tr>
  <tr>
    <td> Moment </td> <td> Moment is a time series foundation model trained through reconstruction tasks. It can be adapted to various downstream tasks by modifying and fine-tuning its lightweight linear head. 
    </td>
  </tr>
  <tr>
    <td> Anomaly Transformer </td> <td> Anomaly Transformer is a neural-based anomaly detection model that reconstructs input time series and computes anomaly scores.
    </td>
  </tr>
  <tr>
    <td> Sub-PCA </td> <td>Sub-PCA is a reconstruction-based statistical method that leverages principal component analysis.</td>
  </tr>
  <tr>
    <td> GPT-4o </td> <td> We also evaluated a representative text-based foundation model, GPT-4o. 
    </td>
  </tr>
</table>

## 🛠️ Settings
The following experimental settings were used for Time-MoE and Moment.
<table>
  <thead>
    <tr>
      <th>Training Setting</th> <th>Description</th>
    </tr>
  </thead>
  <tr>
    <td> Zero-shot </td> <td> The pretrained model was used without additional training or adaptation. </td>
  </tr>
  <tr>
    <td> Out-domain fine-tuning </td> <td> The pretrained model was fine-tuned from out-domain data. </td>
  </tr>
  <tr>
    <td> In-domain fine-tuning </td> <td> The pretrained model was fine-tuned from the target domain data. </td>
  </tr>
  <tr>
    <td> RATFM </td> <td> We retrained the baseline model from the same data as Out-domain FT.</td>
  </tr>
  <tr>
    <td> RATFM w/o training </td> <td> The same input as that of RATFM was fed to a Zero-shot model. This setting aimed to examine the capability of vanilla time series foundation models to use examples. </td>
  </tr>
</table>

Because the existing methods do not perform fine-tuning, we applied the following common experimental settings to Anomaly Transformer and Sub-PCA.

<table>
  <thead>
    <tr>
      <th>Training Setting</th> <th>Description</th>
    </tr>
  </thead>
  </tr>
  <tr>
    <td> Time Series-wise </td> <td> This setting, which has been used in many previous studies on anomaly detection, trains a separate model for each time series, even when they belong to the same domain. </td>
  </tr>
  <tr>
    <td> In-domain </td> <td> We conducted model training and evaluation under the same experimental setting for a fair comparison with In-domain FT. 
    </td>
  </tr>
</table>

## 🧑‍💻 Usage

### 🗂 Dataset Setup
1. **Download the UCR Anomaly Archive**

   Get the dataset from [this GitHub repository](https://github.com/thuml/Large-Time-Series-Model/tree/main/scripts/anomaly_detection) and place it into `data/UCR_Anomaly_Archive/` folder.
2. **Edit the Config File**

   Open `common/config.yaml` and set the correct path to the dataset folder.


### 🔍 Anomaly Detection with Time-MoE and Moment
### 🚀 **RATFM**

1. **Prepare Dataset**
   
   Move into the script directory and generate the required `.jsonl` files.
   ```bash
   cd scripts/{model}
   # For standard RATFM setting
   python generate_ratfm_train_jsonl.py
   python generate_ratfm_eval_jsonl.py
   cd script
   python merge_other_domains_jsonl.py
   ```
    > 💡 **Note**  
    > If you are running experiments for `moment` with **RATFM (reconstruction)** setting,  
    > please run the following instead:
    >
    > ```bash
    > python generate_ratfm_reconstruction_train_jsonl.py
    > python generate_ratfm_reconstruction_eval_jsonl.py
    > ```
    
2. **Edit the Config File**
   
   Update the config file `models/{model}/configs/config_{setting}.yaml`.

3. **Retrain the Model**
   
   Once dataset and config are ready, run:
    ```bash
    cd models/{model}/settings/{setting}
    python run_{setting}.py
    ```
   This will retrain the model and save the model weights to the directory specified by `save_base` of the config file.
      
4. **Run Anomaly Detection**
   
   Use the retrained model to evaluate anomaly detection:
    ```bash
    python eval_{setting}.py
    ```
    Evaluation results are saved as:
    * `L1Loss/`: Results using raw anomaly score
    * `L1Loss_SMA/`: Results using smoothed scores (simple moving average)
    
5.  **Summarize the Results**

    Edit the config file `script/metrics_summary_config.yaml` and run the summarization script:
    ```bash
    cd script
    python summarize_results.py
    ```
    
    This script calculates and outputs the average VUS-ROC, VUS-PR, and F1 score across all datasets.

### 🚀 **Zero-shot**
* Follow steps 2, 4, and 5 from the procedure above.
* Note: Skip step 3 since no retraining is needed.

### 🚀 **In-domain FT**
* (Only for `Time-MoE`) Instead of step 1, move into the script directory and generate the required `.jsonl` files:
```bash
cd scripts/{model}
python generate_domain_group_jsonl.py
```
* Follow steps 2 through 5 from the procedure above.

### 🚀 **Out-domain FT**
* (Only for `Time-MoE`) Instead of step 1, move into the script directory and generate the required `.jsonl` files:
```bash
cd scripts/{model}
python generate_exclude_domain_jsonl.py
```
* Follow steps 2 through 5 from the procedure above.

### 🔍 Anomaly Detection with Anomaly Transformer and Sub-PCA
### 🚀 **Time Series-wise** and **In-domain**
1. **Train the Model and Run Anomaly Detection**
   
   Run the main script:
    ```bash
    cd models/{model}/settings/{setting}
    python main_{setting}.py
    ```
    Evaluation results are saved in the folders `L1Loss/` and `L1Loss_SMA/`.
    
2.  **Summarize the Results**

    Edit the config file `script/metrics_summary_config.yaml` and run the summarization script:
    ```bash
    cd script
    python summarize_results.py
    ```
    
    This script calculates and outputs the average VUS-ROC, VUS-PR, and F1 score across all datasets.

### 🔍 Anomaly Detection with GPT-4o
### 🚀 **Zero-shot**
1. **Prepare Dataset**
   
   Move into the script directory and generate the required `.dkl` files.
    ```bash
    cd models/gpt-4o
    python preprocess_ucr_train.py
    python preprocess_ucr_eval.py
    ```   
2. **Set API Credentials**
   
   Write your OpenAI API key in the file `models/gpt-4o/credentials.yml`
   
3. **Run Anomaly Detection**
   
   Run the main scripts:
    ```bash
    cd models/gpt-4o/src
    python openai_api.py
    python result_agg.py
    ```
4.  **Summarize the Results**
   
    Edit the config file `script/metrics_summary_config.yaml` and run the summarization script:
    ```bash
    cd script
    python summarize_results.py
