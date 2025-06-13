import os
import torch
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from mydataset_std import MyDataset
from sklearn.model_selection import LeaveOneOut
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "E:/learn_pytorch/Aging_modes_estimation/data/data_SOH_ZQY"
file_paths = [os.path.join(data_dir, fname) for fname in ["90_1.xlsx", "90_2.xlsx", "90_3.xlsx", "90_5.xlsx"]]
input_columns = ["F5", "F567"]
output_columns = ["SOH"]

ratio_early = 1
ratio_val = 0.2
times = 10
number = 1
sta = True

model_names = ['SVM', 'Linear', 'GPR']

def train_and_test_sklearn_model(model, train_dataset, test_dataset):
    X_train = train_dataset.inputs
    y_train = train_dataset.outputs
    X_test = test_dataset.inputs
    y_test = test_dataset.outputs
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test).reshape(-1, 1)
    file_path = test_dataset.file_paths[0]
    std_params = test_dataset.std_params[file_path]
    y_mean = std_params["output_mean"].values
    y_std = std_params["output_std"].values
    y_pred_orig = y_pred * y_std + y_mean
    y_test_orig = y_test * y_std + y_mean
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2 = r2_score(y_test_orig, y_pred_orig)
    test_loss_avg = mean_squared_error(y_test_orig, y_pred_orig)

    return test_loss_avg, rmse, r2, y_pred_orig, y_test_orig

for model_name_str in model_names:
    print(f"\n\n======= Training with Model: {model_name_str} =======")
    all_runs_test_stats = []

    for run in range(1, times + 1):
        print(f"\n--- Run {run}/{times} ---")
        results_dir = f"results/90_{model_name_str}/times_{run}"
        os.makedirs(results_dir, exist_ok=True)

        run_exp_name = f"run{run}"
        results_name = f"{results_dir}/{run_exp_name}"

        loo = LeaveOneOut()
        fold_results = []
        test_metrics = []

        for fold, (train_file_idx, test_file_idx) in enumerate(loo.split(file_paths)):
            print(f"\n--- Fold {fold + 1}/4 ---")
            file_paths_train = [file_paths[i] for i in train_file_idx]
            file_paths_test = [file_paths[i] for i in test_file_idx]

            train_dataset = MyDataset(file_paths_train, input_columns, output_columns,ratio_early=ratio_early, standardize=sta)
            test_dataset = MyDataset(file_paths_test, input_columns, output_columns,ratio_early=1.0, standardize=sta)

            # 构造模型
            if model_name_str == 'SVM':
                model = SVR(kernel='rbf', C=10, epsilon=0.01)
            elif model_name_str == 'Linear':
                model = LinearRegression()
            elif model_name_str == 'GPR':
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(
                    noise_level=1e-2)
                model = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, normalize_y=True, n_restarts_optimizer=10)
            else:
                raise ValueError("Unknown model type")

            test_loss_avg, rmse_SOH, r2_SOH, test_predictions, test_labels = train_and_test_sklearn_model(
                model, train_dataset, test_dataset)

            fold_results.append({
                'fold': fold + 1,
                'test_loss_avg': test_loss_avg,
                'rmse_SOH': rmse_SOH,
                'r2_SOH': r2_SOH
            })

            cycle_num = np.arange(1, len(test_labels) + 1).reshape(-1, 1)
            results_df_test = pd.DataFrame(
                data=np.hstack([cycle_num, test_predictions, test_labels, test_predictions - test_labels]),
                columns=["cyc_number"] + [f"Predicted_{col}" for col in output_columns] +
                        [f"True_{col}" for col in output_columns] +
                        [f"Error_{col}" for col in output_columns])
            results_df_test.to_excel(f"{results_name}_fold{fold + 1}_test_results.xlsx", index=False)

            test_metrics.append({
                'test_loss_avg': test_loss_avg,
                'rmse_SOH': rmse_SOH,
                'r2_SOH': r2_SOH
            })

        test_metrics_df = pd.DataFrame(test_metrics)
        test_stats = {
            'test_loss_avg': test_metrics_df['test_loss_avg'].mean(),
            'rmse_SOH': test_metrics_df['rmse_SOH'].mean(),
            'r2_SOH': test_metrics_df['r2_SOH'].mean(),
            'run': run
        }
        all_runs_test_stats.append(test_stats)

        pd.DataFrame(fold_results).to_excel(f"{results_name}_cv_results.xlsx", index=False)
        pd.DataFrame([test_stats]).to_excel(f"{results_name}_test_stats.xlsx", index=False)

    summary_df = pd.DataFrame(all_runs_test_stats).set_index('run')
    mean_row = summary_df.mean().to_frame().T
    mean_row.index = ['mean']
    std_row = summary_df.std().to_frame().T
    std_row.index = ['std']
    summary_df_all = pd.concat([summary_df, mean_row, std_row])
    summary_df_all.to_excel(f"results/90_{model_name_str}/all_runs_summary_{number}.xlsx")

    print(f"\n>>> Finished model {model_name_str}. Results saved in: results/{model_name_str}")
