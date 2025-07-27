from model import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertConfig

import os
import copy
import numpy as np
import argparse
import pytorch_lightning as pl


# 全局非零比例计算函数
def calculate_non_zero_ratio(data):
    """计算数据集中目标值的非零比例"""
    total_count = 0
    non_zero_count = 0

    # 遍历所有批次计算非零比例
    for x, mask, y in data_loader(data['x_train'], data['mask_train'], shuffle=True):
        non_zero_count += torch.sum(y != 0).item()
        total_count += torch.numel(y)

    non_zero_ratio = non_zero_count / (total_count + 1e-8)
    print(f"计算全局非零比例: {non_zero_ratio:.4f} (非零数量: {non_zero_count}, 总数: {total_count})")
    return non_zero_ratio


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=False, help='Resume training from checkpoint')
parser.add_argument('--ckpt_path', type=str, default="logs/lightning_logs/version_28/checkpoints/last.ckpt",help='Path to checkpoint file')
args = parser.parse_args()



class DealDataset(Dataset):
    def __init__(self, x_data, mask_data):
        self.x_data = x_data[..., [0, 1, 2, 3, 4, 5, 6]]  # 6.20 512长度 V4 加入类似归一化
        self.mask_data = mask_data
        self.y_data = x_data[..., 7]
        self.len = x_data.shape[0]


    def __getitem__(self, index):
        return self.x_data[index], self.mask_data[index], self.y_data[index]

    def __len__(self):
        return self.len



def data_loader(data_x, data_mask, shuffle=True):
    data = DealDataset(data_x, data_mask)
    loader = DataLoader(
        dataset=data,
        batch_size=32,
        shuffle=shuffle,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    return loader



class DataModule(pl.LightningDataModule):
    def __init__(self, train_data=None, test_data=None, verbose=True):
        super().__init__()

        self.data = train_data
        self.test_data = test_data
        self.verbose = verbose

    # 在初始化时计算全局权重
        self.global_weight = self.calculate_global_weight()

    def calculate_global_weight(self):
        """计算全局非零权重"""
        non_zero_ratio = calculate_non_zero_ratio(self.data)
        global_weight = 1.0 / max(non_zero_ratio, 1e-8)
        print(f"计算全局权重: {global_weight:.2f} (非零比例: {non_zero_ratio:.4f})")
        return global_weight

    def train_dataloader(self):
        return data_loader(self.data['x_train'], self.data['mask_train'])

    def val_dataloader(self):
        return data_loader(self.data['x_valid'], self.data['mask_valid'], False)

    def test_dataloader(self):
        return data_loader(self.test_data['x_test'], self.test_data['mask_test'], False)

    def __repr__(self):
        return ()


def beta_scheduler(step, warmup=6000, max_beta=0.9):
    return min(max_beta, step / warmup * max_beta)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor



class MyLightningModule(pl.LightningModule):
    def __init__(self,
                 config,  # 接收配置而不是完整模型
                 learning_rate=4e-4,  # 256 size 64 4e-4  # 512 size 32 4e-4
                 weight_decay=1e-2
            ):

        super().__init__()
        self.config = BertConfig(**config)
        self.save_hyperparameters("config")  # 保存所有参数
        self.automatic_optimization = False


        self.model = MidiBertLM(MidiBert(self.config))
        self.midibert = self.model.midibert
        #self.global_weight = global_weight


        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.L1Loss(reduction="none")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.MAX_LEN = 256 # 模型输入的序列长度
        self.MASK_WORD = [1, 1, 1, 0, 1, 1, 1]



    def forward(self, x, attn_masks):
        return self.model(x, attn_masks)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=6000, max_iters=29600) # 1500 14800 #256  # V4 采用 3000 warmup
        return optimizer

    def compute_loss(self, predict, target, beta=0.0):
        # mask 构造

        non_zero_mask = (target != 0).float()
        pred_mask = (predict.squeeze(-1).detach() > -1).float()
        target_mask = non_zero_mask
        valid_mask = torch.clamp(pred_mask + target_mask, 0, 1)

        # 归一化 target
        predict_n = predict.squeeze(-1)
        target_n = 2 * (target.float() - 157) / (923 - 157) - 1

        # 计算 MSE

        reg_loss = F.mse_loss(predict_n, target_n, reduction='none')

        # 两部分损失：
        # 1) 真实 mask（target_mask）部分，权重 1 - beta
        loss_true = (reg_loss * target_mask).sum() / (target_mask.sum() + 1e-9)

        # 2) 并集 mask（valid_mask）部分，权重 beta
        loss_pred = (reg_loss * valid_mask).sum() / (valid_mask.sum() + 1e-9)

        # 加权总损失
        reg_loss_value = (1 - beta) * loss_true + beta * loss_pred

        return reg_loss_value


    def compute_accuracy(self, predict, target):
        # 处理预测值：仅保留第一个预测（即索引0），并将其四舍五入
        data_pre = (923 - 157) * (predict.squeeze(-1) + 1) / 2 + 157

        valid_mask = (target != 0).float()


        # Convert to float masks for multiplication
        temp = torch.round(data_pre)

        # 比较预测值和目标值的第4个通道
        correct = (temp == target).float() * valid_mask
        acc = torch.sum(correct) / torch.sum(valid_mask)

        # 返回单个准确率值（保持在列表中）
        return acc

    def get_mask_ind(self):
        mask_ind = random.sample([i for i in range(self.MAX_LEN)], round(self.MAX_LEN))
        mask50 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
        left = list((set(mask_ind) - set(mask50)))
        rand25 = random.sample(left, round(len(mask_ind) * 0.2))
        cur25 = list(set(left) - set(rand25))
        return mask50, rand25, cur25


    def create_masks(self, inputs):
        mask50, rand25, cur25 = self.get_mask_ind()
        input_seqs = copy.deepcopy(inputs)
        self.loss_mask = torch.ones(inputs.shape)
        for b in range(input_seqs.shape[0]):
            for i in mask50:
                mask_word = torch.tensor(self.MASK_WORD).to(self.device)
                input_seqs[b][i] *= mask_word
            for i in rand25:
                rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                input_seqs[b][i][3] = rand_word
        self.loss_mask = self.loss_mask.to(self.device)
        return input_seqs.long()

    def training_step(self, batch, batch_idx):
        inputs, masks, targets = batch

        targets = torch.clamp(targets, max=923)

        beta = beta_scheduler(self.global_step)  # 新增 beta

        inputs_masked = self.create_masks(inputs)
        outputs = self(inputs_masked, masks)

        loss = self.compute_loss(outputs, targets, beta=beta)  # 传入 beta
        accuracy = self.compute_accuracy(outputs, targets)

        opt = self.optimizers()

        # 反向传播主损失
        opt.zero_grad()
        self.manual_backward(loss, retain_graph=True)

        # 计算梯度范数（GradNorm）
        with torch.autograd.set_detect_anomaly(True):
            dl = torch.autograd.grad(
                loss,
                self.model.mask_lm.proj.parameters(),
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            gw = torch.norm(dl) if dl is not None else torch.tensor(0.0).to(self.device)

        # GradNorm附加项
        gw_avg = gw.detach()
        constant = gw_avg * 1.0  # 单任务时设为1.0
        gradnorm_loss = torch.abs(gw - constant)

        self.manual_backward(gradnorm_loss)
        opt.step()
        self.lr_scheduler.step()

        self.log_dict({
            'train_loss': loss.item(),
            'train_acc': accuracy.item(),
            'beta': beta
        }, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, masks, targets = batch

        targets = torch.clamp(targets, max=923)

        outputs = self(inputs, masks)

        val_loss_combined = self.compute_loss(outputs, targets, beta=0.9)
        val_loss_true = self.compute_loss(outputs, targets, beta=0.0)

        val_acc = self.compute_accuracy(outputs, targets)

        self.log("val_loss_true_mask", val_loss_true, prog_bar=False)
        self.log("val_loss", val_loss_combined, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        return val_loss_combined

    def test_step(self, batch, batch_idx):
        inputs, masks, targets = batch

        targets = torch.clamp(targets, max=923)

        outputs = self(inputs, masks)

        test_loss_true = self.compute_loss(outputs, targets, beta=0.0)
        test_loss_combined = self.compute_loss(outputs, targets, beta=0.9)

        test_acc = self.compute_accuracy(outputs, targets)

        self.log("test_loss_true_mask", test_loss_true, prog_bar=False)
        self.log("test_loss", test_loss_combined, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

        return test_loss_combined


if __name__ == "__main__":

    save_dir = "./processed_data_small"  # 数据存储目录
    MAX_LEN = 256


    # 构建掩码数据路径
    data_filename = f"train_data.npy"  # 关键修改：直接加载掩码文件
    data_path = os.path.join(save_dir, data_filename)
    masks_filename = f"train_mask.npy"  # 关键修改：直接加载掩码文件
    masks_path = os.path.join(save_dir, masks_filename)
    x_train = np.load(data_path)
    mask_train = np.load(masks_path)

    val_filename = f"val_data.npy"
    val_path = os.path.join(save_dir, val_filename)
    val_data_path = os.path.join(save_dir, f"val_data.npy")
    val_masks_path = os.path.join(save_dir, f"val_mask.npy")
    x_valid = np.load(val_data_path)
    mask_val = np.load(val_masks_path)


    # 假设你已经有了 x_train 和 y_train
    n_samples = x_train.shape[0]  # 样本数量
    val_n_samples = x_valid.shape[0]


    # 2. 封装成 DataModule 要求的字典格式
    data = {
        'x_train': x_train,
        'mask_train': mask_train,

        'x_valid': x_valid,
        'mask_valid': mask_val
    }

    # 初始化DataModule以计算全局权重
    datamodule = DataModule(train_data=data)
    global_weight = datamodule.global_weight
    print(f"使用全局权重: {global_weight:.2f}")

    # tensorboard --logdir=D:\project\dataset\maestro-v3.0.0\logs\exp

    # 检查点回调配置
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # 可选：监控验证损失（如不需要可设为None）
        save_top_k=-1,  # 保存所有符合条件的检查点
        every_n_epochs=5,  # 关键：每5个epoch保存一次
        filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}",
        auto_insert_metric_name=False,
        save_last=True,  # 额外保存最新模型（可选）
    )

    # 替换原有CSVLogger
    from pytorch_lightning.loggers import TensorBoardLogger

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs= 200,
        accelerator="gpu",
        devices=1,
        precision="32",
        logger=TensorBoardLogger(
            'logs/'
        ),
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch')  # 只在epoch结束时记录LR
        ],
    )


    # 模型初始化逻辑
    if args.resume:
        print(f"\n=== 从检查点恢复训练: {args.ckpt_path} ===")
        configuration = BertConfig(
            max_position_embeddings=MAX_LEN,
            position_embedding_type='relative_key_query',
            hidden_size=256,    # 128
            num_hidden_layers=6,    # 4
            num_attention_heads=8,    # 4
            intermediate_size=1024,    #128
            attn_implementation="eager"
        )
        config_dict = configuration.to_dict()
        lightning_model = MyLightningModule.load_from_checkpoint(
            args.ckpt_path
        )
    else:
        print("\n=== 初始化新模型 ===")
        configuration = BertConfig(
            max_position_embeddings=MAX_LEN,
            position_embedding_type='relative_key_query',
            hidden_size=256,    # 128
            num_hidden_layers=6,    # 4
            num_attention_heads=8,    # 4
            intermediate_size=1024,    #128
            attn_implementation="eager"
        )
        config_dict = configuration.to_dict()  # 转换为字典
        lightning_model = MyLightningModule(
            config_dict
        )


    print("\n=== 启动完整训练 ===")

    trainer.fit(
        lightning_model,
        datamodule=DataModule(data)
    )






