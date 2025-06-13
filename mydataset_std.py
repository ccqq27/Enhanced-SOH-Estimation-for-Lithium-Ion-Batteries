import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, file_paths, input_columns, output_columns, ratio_early, standardize=True):
        self.file_paths = file_paths
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.standardize = standardize
        # 存储标准化参数
        self.std_params = {}
        self.file_labels = []
        # 读取多个文件的数据
        data_list = []
        for file_idx, file_path in enumerate(self.file_paths):
            data = pd.read_excel(file_path)
            split_idx = int(len(data) * ratio_early)
            data = data.iloc[:split_idx, :]
            assert all(col in data.columns for col in input_columns + output_columns), "The specified column name does not exist"

            input_mean = data[input_columns].mean()
            input_std = data[input_columns].std()
            output_mean = data[output_columns].mean()
            output_std = data[output_columns].std()

            self.std_params[file_path] = {"input_mean": input_mean, "input_std": input_std,
                                          "output_mean": output_mean, "output_std": output_std}

            if self.standardize:
                data[input_columns] = (data[input_columns] - input_mean) / input_std
                data[output_columns] = (data[output_columns] - output_mean) / output_std

            data_list.append(data)
            self.file_labels.extend([file_idx] * len(data))

            self.data = pd.concat(data_list, ignore_index=True)

            self.inputs = self.data[self.input_columns].values.astype(np.float32)
            self.outputs = self.data[self.output_columns].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_idx = self.file_labels[idx]
        inputs = torch.tensor(row[self.input_columns].values, dtype=torch.float32)
        outputs = torch.tensor(row[self.output_columns].values, dtype=torch.float32)
        return inputs, outputs, file_idx

    def inverse_transform(self, inputs, outputs, file_indices):
        batch_size = inputs.shape[0]
        orig_inputs = []
        orig_outputs = []
        for i in range(batch_size):
            file_idx = file_indices[i].item()
            file_path = self.file_paths[file_idx]
            std_params = self.std_params[file_path]
            input_mean_vals = torch.tensor(std_params["input_mean"].values, dtype=torch.float32)
            input_std_vals = torch.tensor(std_params["input_std"].values, dtype=torch.float32)
            output_mean_vals = torch.tensor(std_params["output_mean"].values, dtype=torch.float32)
            output_std_vals = torch.tensor(std_params["output_std"].values, dtype=torch.float32)
            orig_input = inputs[i].clone()
            orig_output = outputs[i].clone()
            if self.standardize:
                orig_input = orig_input * input_std_vals + input_mean_vals
                orig_output = orig_output * output_std_vals + output_mean_vals
            orig_inputs.append(orig_input.unsqueeze(0))
            orig_outputs.append(orig_output.unsqueeze(0))
        return torch.cat(orig_inputs, dim=0), torch.cat(orig_outputs, dim=0)
