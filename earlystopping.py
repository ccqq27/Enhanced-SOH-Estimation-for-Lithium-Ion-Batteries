import torch

class EarlyStopping:
    def __init__(self, patience=100, delta=0, verbose=True, model_name="best_model.pth"):
        self.patience = patience  # 允许的停滞周期数
        self.delta = delta  # 视为改善的最小变化量
        self.verbose = verbose  # 是否打印提示信息
        self.counter = 0  # 当前停滞计数
        self.best_score = None  # 最佳分数记录
        self.early_stop = False  # 停止标志
        self.best_loss = float('inf')
        self.model_name = model_name  # 动态模型名称

    def __call__(self, val_loss, model):
        score = -val_loss

        # 首次初始化
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)

        # 如果当前损失未改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            # if self.verbose:
            #     print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            # 检查是否达到停止条件
            if self.counter >= self.patience:
                self.early_stop = True
        # 如果损失改善
        else:
            self.best_loss = val_loss
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # 保存最佳模型
        # if self.verbose:
        #     print(f'Validation loss decreased ({self.best_loss:.5f} --> {val_loss:.5f}). Saving model to {self.model_name}...')
        torch.save(model.state_dict(), self.model_name)
        self.best_loss = val_loss

    def get_best_loss(self):
        return self.best_loss