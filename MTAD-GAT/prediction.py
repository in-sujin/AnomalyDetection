import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, classification_report
from eval_methods import *  # Adjust imports according to your setup
from utils import *  # Adjust imports according to your setup

def ROC(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def cal_metrics(labels, outputs, window_size):
    labels = np.array(labels)
    outputs = np.array(outputs)

    
    num_windows = len(labels) - window_size + 1
    if num_windows <= 0:
        raise ValueError("Window size is too large for the given labels.")

    windows_labels = [labels[i:i + window_size] for i in range(num_windows)]
    
    y_test = [1.0 if (np.sum(window) > 0) else 0.0 for window in windows_labels]
    
    # Ensure the length of y_test matches the length of outputs
    if len(y_test) != len(outputs):
        # Adjust length of outputs to match y_test
        outputs = outputs[:len(y_test)]
        
    # predicted output
    y_pred = outputs

    # threshold calculation using ROC
    threshold = ROC(y_test, y_pred)

    # apply threshold to predictions
    y_pred_ = [1.0 if (score >= threshold) else 0.0 for score in y_pred]

    # Print classification report
    print(classification_report(y_test, y_pred_))
    return classification_report(y_test, y_pred_, output_dict=True)

class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies
    :param summary_file_name: File name for saving summary results
    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return: DataFrame with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                y_hat, _ = self.model(x)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y), dim=1)
                _, window_recon = self.model(recon_x)

                preds.append(y_hat.detach().cpu().numpy())
                recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        actual = values[self.window_size:].detach().cpu().numpy()  # Convert Tensor to numpy

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df_dict = {}
        for i in range(preds.shape[1]):
            df_dict[f"Forecast_{i}"] = preds[:, i]
            df_dict[f"Recon_{i}"] = recons[:, i]
            df_dict[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
                (recons[:, i] - actual[:, i]) ** 2)

            # Skipping scaling
            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores

        return df

    def predict_anomalies(self, test, true_anomalies, load_scores=False, 
                          save_output=True, metrics=True):
        """ Predicts anomalies

        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        """

        if load_scores:
            print("Loading anomaly scores")

            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            test_pred_df = self.get_score(test)
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            if metrics:
                # Calculate metrics using the cal_metrics function
                metrics = cal_metrics(
                    labels=true_anomalies,
                    outputs=test_anomaly_scores,
                    window_size=self.window_size
                )

                # Save results
                with open(f"{self.save_path}/{self.summary_file_name}", "w") as f:
                    json.dump(metrics, f, indent=2)

                # Save anomaly predictions
                if save_output:
                    test_pred_df["A_True_Global"] = true_anomalies
            else:
                return test_anomaly_scores
                test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")
