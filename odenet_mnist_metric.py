import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['resnet', 'odenet'], default='odenet')
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--use_adjoint', type=eval, default=False, choices=[True, False])
    parser.add_argument('--down_method', type=str, default='conv', choices=['conv', 'res'])
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--augment', type=eval, default=True, choices=[True, False])
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--val_bs', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default='./exp1')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--cuda_id', type=int, default=0)
    return parser.parse_args()

args = parse_args()

if args.use_adjoint:
    from torchdiffeq import odeint_adjoint as odeint_func
else:
    from torchdiffeq import odeint as odeint_func

def one_by_one_conv(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

def triple_conv(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)

def group_norm_layer(channels):
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

class BasicResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, downsample_layer=None):
        super().__init__()
        self.norm1 = group_norm_layer(in_c)
        self.norm2 = group_norm_layer(out_c)
        self.conv1 = triple_conv(in_c, out_c, stride)
        self.conv2 = triple_conv(out_c, out_c)
        self.downsample = downsample_layer
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(x))
        if self.downsample:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + residual

class TimeConcatConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=0, transpose=False):
        super().__init__()
        conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_layer(c_in + 1, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

    def forward(self, t, x):
        t_channel = torch.ones_like(x[:, :1, :, :]) * t
        combined = torch.cat([t_channel, x], dim=1)
        return self.conv(combined)

class ODEFunction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = group_norm_layer(dim)
        self.norm2 = group_norm_layer(dim)
        self.norm3 = group_norm_layer(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = TimeConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = TimeConcatConv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.nfe_count = 0

    def forward(self, t, x):
        self.nfe_count += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super().__init__()
        self.func = odefunc
        self.integration_times = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_times = self.integration_times.type_as(x)
        out = odeint_func(self.func, x, self.integration_times, rtol=args.tolerance, atol=args.tolerance)
        return out[1]

    @property
    def nfe(self):
        return self.func.nfe_count

    @nfe.setter
    def nfe(self, val):
        self.func.nfe_count = val

class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def infinite_data_loader(iterable):
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(iterable)

def lr_schedule(batch_size, denom, batches_per_epoch, boundaries, decays):
    init_lr = args.learning_rate * batch_size / denom
    steps = [int(batches_per_epoch * ep) for ep in boundaries]
    lr_values = [init_lr * d for d in decays]
    def lr_fn(iteration):
        for i, step in enumerate(steps):
            if iteration < step:
                return lr_values[i]
        return lr_values[-1]
    return lr_fn

def acc_calc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            correct += (preds.cpu() == target).sum().item()
            total += target.size(0)
    return correct / total

def count_params(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

def setup_logger(log_file, code_file, package_files=[], console=True, file_logging=True, debug=False):
    log = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    log.setLevel(level)
    if file_logging:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        log.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        log.addHandler(ch)
    log.info(f'Code file: {code_file}')
    with open(code_file, 'r') as f:
        log.info(f.read())
    for pf in package_files:
        log.info(f'Package file: {pf}')
        with open(pf, 'r') as f:
            log.info(f.read())
    return log

def evaluate_metrics(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    total_correct = 0
    targets_all = []
    probs_all = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs.cpu().numpy())
            targets_all.append(labels.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = total_correct / len(loader.dataset)
    targets_np = np.concatenate(targets_all)
    probs_np = np.concatenate(probs_all)
    try:
        auroc_score = roc_auc_score(np.eye(10)[targets_np], probs_np, average='macro', multi_class='ovr')
    except Exception:
        auroc_score = float('nan')
    return avg_loss, acc, auroc_score

if __name__ == '__main__':
    mkdir_if_missing(args.out_dir)
    logger = setup_logger(os.path.join(args.out_dir, 'train.log'), os.path.abspath(__file__), debug=args.debug_mode)
    logger.info(vars(args))
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')

    if args.down_method == 'conv':
        down_layers = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1),
            group_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            group_norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        ]
    else:
        down_layers = [
            nn.Conv2d(1, 64, 3, 1),
            BasicResidualBlock(64, 64, stride=2, downsample_layer=one_by_one_conv(64, 64, 2)),
            BasicResidualBlock(64, 64, stride=2, downsample_layer=one_by_one_conv(64, 64, 2)),
        ]

    if args.model_type == 'odenet':
        feature_layers = [ODEBlock(ODEFunction(64))]
    else:
        feature_layers = [BasicResidualBlock(64, 64) for _ in range(6)]

    fc_layers = [
        group_norm_layer(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        FlattenLayer(),
        nn.Linear(64, 10)
    ]

    model = nn.Sequential(*down_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info(f'Total trainable params: {count_params(model)}')

    train_loader = DataLoader(
        dsets.MNIST('.data/mnist', train=True, download=True,
                    transform=T.Compose([
                        T.RandomCrop(28, padding=4) if args.augment else T.ToTensor(),
                        T.ToTensor()
                    ]) if args.augment else T.ToTensor()),
        batch_size=args.train_bs, shuffle=True, num_workers=2, drop_last=True
    )

    eval_loader = DataLoader(
        dsets.MNIST('.data/mnist', train=True, download=True, transform=T.ToTensor()),
        batch_size=args.val_bs, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        dsets.MNIST('.data/mnist', train=False, download=True, transform=T.ToTensor()),
        batch_size=args.val_bs, shuffle=False, num_workers=2, drop_last=True
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss_function = nn.CrossEntropyLoss().to(device)

    infinite_train_iter = infinite_data_loader(train_loader)
    steps_per_epoch = len(train_loader)
    lr_function = lr_schedule(args.train_bs, 128, steps_per_epoch, [60, 100, 140], [1, 0.1, 0.01, 0.001])

    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_correct = 0
        n_samples = 0
        start_time = time.time()

        for batch_idx in range(steps_per_epoch):
            iteration = (epoch - 1) * steps_per_epoch + batch_idx
            lr_now = lr_function(iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_now

            inputs, labels = next(infinite_train_iter)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            n_samples += inputs.size(0)

            if batch_idx % 100 == 0:
                logger.info(f'Epoch {epoch} Iter {batch_idx} Loss {loss.item():.4f} LR {lr_now:.6f}')

        epoch_duration = time.time() - epoch_start
        avg_train_loss = total_loss / n_samples
        train_acc = total_correct / n_samples

        logger.info(f'Epoch {epoch} Train Loss: {avg_train_loss:.4f} Accuracy: {train_acc:.4f} Time: {epoch_duration:.2f}s')

        val_loss, val_acc, val_auroc = evaluate_metrics(model, eval_loader, loss_function)
        logger.info(f'Epoch {epoch} Validation Loss: {val_loss:.4f} Accuracy: {val_acc:.4f} AUROC: {val_auroc:.4f}')

        model.eval()
        latency_times = []
        with torch.no_grad():
            for val_data, _ in eval_loader:
                val_data = val_data.to(device)
                start = time.time()
                _ = model(val_data)
                end = time.time()
                latency_times.append(end - start)
        avg_latency = np.mean(latency_times) / val_data.size(0)  # per sample
        logger.info(f'Average inference latency per sample (val set): {avg_latency*1000:.4f} ms')

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))
            logger.info(f'Best model saved at epoch {epoch} with accuracy {best_accuracy:.4f}')

    test_loss, test_acc, test_auroc = evaluate_metrics(model, test_loader, loss_function)
    logger.info(f'Test set metrics — Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, AUROC: {test_auroc:.4f}')
