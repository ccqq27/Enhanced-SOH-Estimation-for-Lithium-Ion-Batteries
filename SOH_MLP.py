import os
import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from model_MLP import MLP
from mydataset_std import MyDataset
from train_val_test_model import train_model, test_model
from sklearn.model_selection import LeaveOneOut

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "E:/learn_pytorch/Aging_modes_estimation/data/data_SOH_ZQY"
file_paths = [os.path.join(data_dir, fname) for fname in ["90_1.xlsx", "90_2.xlsx", "90_3.xlsx", "90_5.xlsx"]]
input_columns = ["F5", "F567"]
output_columns = ["SOH"]

input_dim = len(input_columns)
output_dim = len(output_columns)
hidden_layers = 5
hidden_dim = 50
dropout = 0
ratio_early = 1
ratio_val = 0.2
batch_size = 256
sta = True
lr = 0.01
times = 10
num_epochs = 1000
patience = 500
number = 9

exp_name = (f'Lr{lr}_layer{hidden_layers}_dim{hidden_dim}_number{number}'
            f'_ratioval{ratio_val}_ratioearly{ratio_early}_dropout{dropout}_sta{sta}')

all_runs_test_stats = []

for run in range(1, times + 1):
    print(f"\n=== Starting Run {run}/{times} ===")
    results_dir = f"results/90_MLP_{ratio_early}_{lr}_{hidden_layers}_{hidden_dim}_inp{input_dim}/times_{run}"
    modelsave_dir = f"model_save/90_MLP_{ratio_early}_{lr}_{hidden_layers}_{hidden_dim}_inp{input_dim}/times_{run}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(modelsave_dir, exist_ok=True)

    run_exp_name = f"run{run}"
    model_name = f"{modelsave_dir}/{run_exp_name}"
    results_name = f"{results_dir}/{run_exp_name}"

    loo = LeaveOneOut()
    fold_results = []
    test_metrics = []

    for fold, (train_file_idx, test_file_idx) in enumerate(loo.split(file_paths)):
        print(f"\n=== Fold {fold + 1}/4 ===")

        file_paths_train = [file_paths[i] for i in train_file_idx]
        file_paths_test = [file_paths[i] for i in test_file_idx]

        train_dataset = MyDataset(file_paths_train, input_columns, output_columns, ratio_early, standardize=True)
        test_dataset = MyDataset(file_paths_test, input_columns, output_columns, ratio_early=1, standardize=True)

        total_len = len(train_dataset)
        val_len = int(total_len * ratio_val)
        train_len = total_len - val_len
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(train_dataset, [train_len, val_len], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = MLP(input_dim, output_dim, layers_num=hidden_layers, hidden_dim=hidden_dim, dropout=dropout)
        model.initialize(method='xavier')
        model = model.to(device)

        fold_model_name = f"{model_name}_fold{fold + 1}"
        model, train_losses, val_losses = train_model(model, train_dataset, train_loader, val_loader,
                                                      num_epochs=num_epochs,
                                                      learning_rate=lr, path_model_save=fold_model_name, device=device,
                                                      patience=patience)

        (test_loss_avg, rmse_SOH, r2_SOH, test_predictions,
         test_labels) = test_model(model=model, test_dataset=test_dataset, test_loader=test_loader, device=device)

        fold_results.append({
            'fold': fold + 1, 'train_loss': train_losses[-1], 'val_loss': val_losses[-1],
            'test_loss_avg': test_loss_avg,
            'rmse_SOH': rmse_SOH, 'r2_SOH': r2_SOH
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
            'rmse_SOH': rmse_SOH, 'r2_SOH': r2_SOH
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
    pd.DataFrame(data=np.hstack([np.vstack(train_losses), np.vstack(val_losses)]),
                 columns=['train_loss', 'val_loss']).to_excel(f"{results_name}_loss_curve.xlsx", index=False)

    print("\n=== Fold Summary ===")
    print(pd.DataFrame(fold_results))
    print("\n=== Test Metrics for This Run ===")
    for key, value in test_stats.items():
        if key != "run":
            print(f"{key}: {value:.4f}")

summary_df = pd.DataFrame(all_runs_test_stats)
summary_df.set_index('run', inplace=True)

mean_row = summary_df.mean().to_frame().T
mean_row.index = ['mean']
std_row = summary_df.std().to_frame().T
std_row.index = ['std']

summary_df_all = pd.concat([summary_df, mean_row, std_row])
summary_save_path = f"results/90_MLP_{ratio_early}_{lr}_{hidden_layers}_{hidden_dim}_inp{input_dim}/all_runs_summary_{number}.xlsx"
summary_df_all.to_excel(summary_save_path)

