import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from earlystopping import EarlyStopping


def train_model(model, train_dataset, train_loader, val_loader, num_epochs, learning_rate,
                path_model_save, device, patience):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2 正则化
    scheduler = MultiStepLR(optimizer, milestones=[num_epochs // 2], gamma=0.1)
    loss_func = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, model_name=path_model_save)

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (inputs_train, outputs_train, file_indices_train) in enumerate(train_loader):
            inputs_train = inputs_train.to(device)
            outputs_train = outputs_train.to(device)
            Output = model(inputs_train)
            _, orig_Output = train_dataset.inverse_transform(inputs_train, Output, file_indices_train)
            orig_inputs_train, orig_outputs_train = train_dataset.inverse_transform(inputs_train, outputs_train, file_indices_train)
            loss_mse_train = loss_func(orig_Output, orig_outputs_train)
            loss_train = loss_mse_train
            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss_train.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        for i, (inputs_val, outputs_val, file_indices_val) in enumerate(val_loader):
            inputs_val = inputs_val.to(device)
            outputs_val = outputs_val.to(device)
            pre_AM = model(inputs_val)
            _, orig_pre_AM = train_dataset.inverse_transform(inputs_val, pre_AM, file_indices_val)
            _, orig_outputs_val = train_dataset.inverse_transform(inputs_val, outputs_val, file_indices_val)
            loss_test = loss_func(orig_pre_AM, orig_outputs_val)
            val_loss += loss_test.item()
        val_losses.append(val_loss / len(val_loader))
        current_val_loss = val_loss / len(val_loader)
        early_stopping(current_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}, Best Validation Loss: {early_stopping.get_best_loss():.5f}")
            break
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch: [{epoch + 1}/{num_epochs}], trainLoss:{train_losses[epoch]:.5f}, valLoss:{val_losses[epoch]:.5f}')
    model.load_state_dict(torch.load(path_model_save))
    return model, train_losses, val_losses

def test_model(model, test_dataset, test_loader, device):
    loss_func = nn.MSELoss()
    model.eval()
    test_loss = 0
    test_predictions = []
    test_labels = []
    for i, (inputs_test, outputs_test, file_indices_test) in enumerate(test_loader):
        inputs_test = inputs_test.to(device)
        outputs_test = outputs_test.to(device)
        pre_AM = model(inputs_test)
        _, orig_pre_AM = test_dataset.inverse_transform(inputs_test, pre_AM, file_indices_test)
        _, orig_outputs_test = test_dataset.inverse_transform(inputs_test, outputs_test, file_indices_test)
        test_predictions.append(orig_pre_AM.detach().cpu().numpy())
        test_labels.append(orig_outputs_test.cpu().numpy())
        loss_test = loss_func(orig_pre_AM, orig_outputs_test)  # 预测与标签之间的损失
        test_loss += loss_test.item()
    test_loss_avg = (test_loss / len(test_loader))
    print(f'\nFinal Test Loss: {test_loss_avg:.5f}')
    test_predictions = np.vstack(test_predictions)
    test_labels = np.vstack(test_labels)

    rmse_SOH = np.sqrt(mean_squared_error(test_labels[:, 0], test_predictions[:, 0]))
    print(f'RMSE_SOH: {rmse_SOH:.4f}')
    r2_SOH = r2_score(test_labels[:, 0], test_predictions[:,0])
    print(f'R-squared_SOH: {r2_SOH:.4f}')
    print(f'\nFinal Test Loss (MSE): {test_loss_avg:.5f}')
    print('--- Errors ---')
    print(f'SOH_RMSE: {rmse_SOH:.4f}, SOH_R2:{r2_SOH:.4f}')

    return (test_loss_avg, rmse_SOH, r2_SOH, test_predictions, test_labels)