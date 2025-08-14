from model_FR import *
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig

import os
import copy
import numpy as np
import argparse
import pytorch_lightning as pl

# Function to calculate the global non-zero ratio
def calculate_non_zero_ratio(data):
    """Compute the proportion of non-zero target values in the dataset"""
    total_count = 0
    non_zero_count = 0

    # Iterate over batches
    for x, mask, y in data_loader(data['x_train'], data['mask_train'], shuffle=True):
        non_zero_count += torch.sum(y != 0).item()
        total_count += torch.numel(y)

    non_zero_ratio = non_zero_count / (total_count + 1e-8)
    print(f"Global non-zero ratio: {non_zero_ratio:.4f} (non-zero: {non_zero_count}, total: {total_count})")
    return non_zero_ratio

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=True, help='Resume training from checkpoint')
parser.add_argument('--ckpt_path', type=str, default="logs/lightning_logs/version_10/checkpoints/epoch_14-val_loss_0.21.ckpt",
                    help='Path to checkpoint file')
args = parser.parse_args()


class DealDataset(Dataset):
    """Custom Dataset for handling input, mask, and target values"""
    def __init__(self, x_data, mask_data):
        self.x_data = x_data[..., [0, 1, 2, 3, 4]]  # Input features
        self.mask_data = mask_data
        self.y_data = x_data[..., [5, 6, 7]]       # Target features
        self.len = x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.mask_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_loader(data_x, data_mask, shuffle=True):
    """Wrapper for DataLoader with batch size and workers"""
    dataset = DealDataset(data_x, data_mask)
    loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=shuffle,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    return loader


class DataModule(pl.LightningDataModule):
    """Lightning DataModule to handle training, validation, and test loaders"""
    def __init__(self, train_data=None, test_data=None, verbose=True):
        super().__init__()
        self.data = train_data
        self.test_data = test_data
        self.verbose = verbose

        # Compute global weight upon initialization
        self.global_weight = self.calculate_global_weight()

    def calculate_global_weight(self):
        """Compute weight based on global non-zero ratio"""
        non_zero_ratio = calculate_non_zero_ratio(self.data)
        global_weight = 1.0 / max(non_zero_ratio, 1e-8)
        print(f"Global weight: {global_weight:.2f} (non-zero ratio: {non_zero_ratio:.4f})")
        return global_weight

    def train_dataloader(self):
        return data_loader(self.data['x_train'], self.data['mask_train'])

    def val_dataloader(self):
        return data_loader(self.data['x_valid'], self.data['mask_valid'], shuffle=False)

    def test_dataloader(self):
        return data_loader(self.test_data['x_test'], self.test_data['mask_test'], shuffle=False)

    def __repr__(self):
        return ()


# Beta scheduling function
def beta_scheduler(step, warmup=6000, max_beta=0.9):
    """Compute dynamic beta for weighted loss"""
    return min(max_beta, step / warmup * max_beta)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warmup for learning rate"""
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
    """Main Lightning module for training MIDI BERT model"""
    def __init__(self, config, learning_rate=4e-4, weight_decay=1e-2):
        super().__init__()
        self.config = BertConfig(**config)
        self.save_hyperparameters("config")  # Save all parameters
        self.automatic_optimization = False

        self.model = MidiBertLM(MidiBert(self.config))
        self.midibert = self.model.midibert

        self.loss_mse = nn.L1Loss(reduction="none")
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.MAX_LEN = 256  # Model input sequence length
        self.MASK_WORD = [1, 1, 1, 0, 1]  # Mask template

    def forward(self, x, attn_masks):
        return self.model(x, attn_masks)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=6000, max_iters=29600)
        return optimizer

    def compute_loss(self, predict, target, beta=0.0):
        """Compute weighted MSE loss with masking"""
        non_zero_mask = (target != 0).float()

        pred_mask = (predict.detach() > -1.0).float()
        target_mask = non_zero_mask
        valid_mask = torch.clamp(pred_mask + target_mask, 0, 1)

        clamp_min_p = round(2437 - (2820 - 2437) / 17.3)
        clamp_min_d = round(157 - (923 - 157) / 17.3)

        target_p = torch.clamp(target[..., 0].float(), min=clamp_min_p)
        target_d = torch.clamp(target[..., 1].float(), min=clamp_min_d)

        target_p_n = 2 * (target_p - 2437) / (2820 - 2437) - 1
        target_d_n = 2 * (target_d - 157) / (923 - 157) - 1
        target_n = torch.stack((target_p_n, target_d_n), dim=-1)

        reg_loss = (predict - target_n) ** 2
        reg_loss_masked_p = (reg_loss[..., 0] * valid_mask[..., 0]).sum() / (valid_mask[..., 0].sum() + 1e-9)
        reg_loss_masked_d = (reg_loss[..., 1] * valid_mask[..., 1]).sum() / (valid_mask[..., 1].sum() + 1e-9)

        loss = (1 - beta) * (
            (reg_loss[..., 0] * target_mask[..., 0]).sum() / (target_mask[..., 0].sum() + 1e-9)
            + (reg_loss[..., 1] * target_mask[..., 1]).sum() / (target_mask[..., 1].sum() + 1e-9)
        ) / 2 + beta * (reg_loss_masked_p + reg_loss_masked_d) / 2

        return loss, reg_loss_masked_p, reg_loss_masked_d

    def compute_accuracy(self, predict, target):
        """Compute accuracy for pitch and duration"""
        data_p = (2820 - 2437) * (predict[..., 0] + 1) / 2 + 2437
        data_d = (923 - 157) * (predict[..., 1] + 1) / 2 + 157
        non_zero_mask_float = (target != 0).float()

        temp_p = torch.round(data_p)
        temp_d = torch.round(data_d)

        correct_p = ((temp_p == target[..., 0]).float() * non_zero_mask_float[..., 0]).sum() / torch.sum(non_zero_mask_float[..., 0])
        correct_d = ((temp_d == target[..., 1]).float() * non_zero_mask_float[..., 1]).sum() / torch.sum(non_zero_mask_float[..., 1])
        acc = (correct_p + correct_d) / 2

        return acc, correct_p, correct_d

    def get_mask_ind(self):
        """Generate mask indices for training"""
        mask_ind = random.sample([i for i in range(self.MAX_LEN)], round(self.MAX_LEN))
        mask50 = random.sample(mask_ind, round(len(mask_ind) * 0.8))
        left = list((set(mask_ind) - set(mask50)))
        rand25 = random.sample(left, round(len(mask_ind) * 0.2))
        cur25 = list(set(left) - set(rand25))
        return mask50, rand25, cur25

    def create_masks(self, inputs):
        """Apply masking to input sequences"""
        mask50, rand25, cur25 = self.get_mask_ind()
        input_seqs = copy.deepcopy(inputs)
        for b in range(input_seqs.shape[0]):
            for i in mask50:
                mask_word = torch.tensor(self.MASK_WORD).to(self.device)
                input_seqs[b][i] *= mask_word
            for i in rand25:
                rand_word = torch.tensor(self.midibert.get_rand_tok()).to(self.device)
                input_seqs[b][i][3] = rand_word
        return input_seqs.long()

    def training_step(self, batch, batch_idx):
        """Single training step with manual optimization and GradNorm"""
        inputs, masks, targets = batch

        # Clamp target duration
        targets[..., [2]] = torch.clamp(targets[..., [2]], max=923)

        # Dynamic beta scheduling
        beta = beta_scheduler(self.global_step)

        # Initialize weights for non-zero targets
        weights = torch.ones_like(targets)
        non_zero_mask = (targets != 0)
        weights[non_zero_mask] = global_weight  # global_weight from DataModule

        # Create masked input sequences
        inputs_masked = self.create_masks(inputs)

        # Forward pass
        outputs = self(inputs_masked, masks)

        # Compute loss (returns total, pitch loss, duration loss)
        loss, loss_p, loss_d = self.compute_loss(outputs, targets[..., [1, 2]], beta=beta)

        # Compute accuracy
        accuracy, _, _ = self.compute_accuracy(outputs, targets[..., [1, 2]])

        # Get optimizer
        opt = self.optimizers()

        # Weighted loss (single-task)
        weighted_loss = loss

        # Backward pass
        opt.zero_grad()
        self.manual_backward(weighted_loss, retain_graph=True)

        # GradNorm calculation (single-task version)
        with torch.autograd.set_detect_anomaly(True):
            dl = torch.autograd.grad(
                loss,
                self.model.mask_lm.proj.parameters(),
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
            gw = torch.norm(dl) if dl is not None else torch.tensor(0.0).to(self.device)

        gw_avg = gw.detach()
        constant = gw_avg * 1.0
        gradnorm_loss = torch.abs(gw - constant)

        # Backward on GradNorm loss
        self.manual_backward(gradnorm_loss)

        # Parameter update
        opt.step()
        self.lr_scheduler.step()

        # Logging
        self.log_dict({
            'train_loss': loss.item(),
            'loss_p': loss_p.item(),
            'loss_d': loss_d.item(),
            'train_acc': accuracy.item(),
            'beta': beta
        }, on_step=False, on_epoch=True, prog_bar=True)

        return weighted_loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        inputs, masks, targets = batch
        targets[..., [2]] = torch.clamp(targets[..., [2]], max=923)

        outputs = self(inputs, masks)

        # Loss using ground-truth mask
        val_loss_true, _, _ = self.compute_loss(outputs, targets[..., [1, 2]], beta=0.0)

        # Combined loss using weighted mask
        val_loss_combined, _, _ = self.compute_loss(outputs, targets[..., [1, 2]], beta=0.9)

        # Compute accuracy
        val_acc, _, _ = self.compute_accuracy(outputs, targets[..., [1, 2]])

        # Logging
        self.log("val_loss_true_mask", val_loss_true, prog_bar=False)
        self.log("val_loss", val_loss_combined, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)

        return val_loss_combined

    def test_step(self, batch, batch_idx):
        """Test step"""
        inputs, masks, targets = batch
        targets[..., [2]] = torch.clamp(targets[..., [2]], max=923)

        outputs = self(inputs, masks)

        test_loss_true, _, _ = self.compute_loss(outputs, targets[..., [1, 2]], beta=0.0)
        test_loss_combined, _, _ = self.compute_loss(outputs, targets[..., [1, 2]], beta=0.9)
        test_acc, _, _ = self.compute_accuracy(outputs, targets[..., [1, 2]])

        self.log("test_loss_true_mask", test_loss_true, prog_bar=False)
        self.log("test_loss_combined_mask", test_loss_combined, prog_bar=True)
        self.log("test_acc", test_acc, prog_bar=True)

        return test_loss_combined


if __name__ == "__main__":
    import psutil

    # Set high process priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    save_dir = "./processed_data_small"
    MAX_LEN = 256

    # Load training and validation data
    x_train = np.load(os.path.join(save_dir, "train_data.npy"))
    mask_train = np.load(os.path.join(save_dir, "train_mask.npy"))

    x_valid = np.load(os.path.join(save_dir, "val_data.npy"))
    mask_val = np.load(os.path.join(save_dir, "val_mask.npy"))

    # Package data for DataModule
    data = {
        'x_train': x_train,
        'mask_train': mask_train,
        'x_valid': x_valid,
        'mask_valid': mask_val
    }

    # Initialize DataModule to compute global weight
    datamodule = DataModule(train_data=data)
    global_weight = datamodule.global_weight
    print(f"Global weight: {global_weight:.2f}")

    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=-1,  # save all checkpoints
        every_n_epochs=5,
        filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}",
        auto_insert_metric_name=False,
        save_last=True
    )

    # TensorBoard logger
    from pytorch_lightning.loggers import TensorBoardLogger

    trainer = pl.Trainer(
        profiler="simple",
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        precision="32",
        logger=TensorBoardLogger('logs/'),
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='epoch')],
    )

    # Model initialization
    if args.resume:
        print(f"\n=== Resuming from checkpoint: {args.ckpt_path} ===")
        lightning_model = MyLightningModule.load_from_checkpoint(args.ckpt_path)
    else:
        print("\n=== Initializing new model ===")
        configuration = BertConfig(
            max_position_embeddings=MAX_LEN,
            position_embedding_type='relative_key_query',
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            attn_implementation="eager"
        )
        config_dict = configuration.to_dict()
        lightning_model = MyLightningModule(config_dict)

    # Start training
    print("\n=== Starting training ===")
    trainer.fit(lightning_model, datamodule=DataModule(data))
