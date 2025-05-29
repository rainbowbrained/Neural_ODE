import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import resample

parser = argparse.ArgumentParser(description='ODE-Net for PhysioNet ECG Classification')
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3, help='Tolerance for ODE solver')
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
parser.add_argument('--save', type=str, default='./physionet_experiment', help='Save directory')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def download_mitdb():
    db_dir = 'mitdb'
    if not os.path.exists(db_dir):
        print("Downloading MIT-BIH Arrhythmia Database...")
        try:
            wfdb.dl_database('mitdb', db_dir)
            print("Download complete!")
        except Exception as e:
            print(f"Failed to download database: {str(e)}")
            raise
    else:
        print("MIT-BIH database already exists, skipping download")
    return db_dir


class PhysioNetECGDataset(Dataset):
    def __init__(self, record_names, db_dir='mitdb', segment_length=250, classes=None):
        self.records = []
        self.labels = []
        self.segment_length = segment_length

        if classes is None:
            self.classes = ['N', 'L', 'R', 'V', 'A']
        else:
            self.classes = classes

        for record_name in record_names:
            try:
                record_path = os.path.join(db_dir, record_name)

                if not all(os.path.exists(f"{record_path}.{ext}") for ext in ['dat', 'hea', 'atr']):
                    print(f"Missing files for record {record_name}, skipping...")
                    continue

                annotation = wfdb.rdann(record_path, 'atr')
                signal, _ = wfdb.rdsamp(record_path, channels=[0])  # Первый канал

                for i in range(len(annotation.symbol)):
                    if annotation.symbol[i] in self.classes:
                        start = annotation.sample[i] - self.segment_length // 2
                        end = start + self.segment_length

                        if start >= 0 and end < len(signal):
                            segment = signal[start:end].flatten()
                            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
                            self.records.append(segment)
                            self.labels.append(annotation.symbol[i])
            except Exception as e:
                print(f"Error processing record {record_name}: {str(e)}")
                continue

        if len(self.records) == 0:
            raise ValueError("No ECG segments were extracted! Check your data and classes.")

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.classes = self.label_encoder.classes_

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        signal = torch.FloatTensor(self.records[idx])
        label = torch.LongTensor([self.labels[idx]])
        return signal.unsqueeze(0), label.squeeze()


class ODEfunc1D(nn.Module):
    def __init__(self, dim):
        super(ODEfunc1D, self).__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(dim + 1, dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim + 1, dim, kernel_size=3, padding=1)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        tt = torch.ones_like(out[:, :1, :]) * t
        out = torch.cat([tt, out], 1)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        tt = torch.ones_like(out[:, :1, :]) * t
        out = torch.cat([tt, out], 1)
        out = self.conv2(out)
        return out


class ODEBlock1D(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock1D, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ResBlock1D(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock1D, self).__init__()
        self.norm1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class RunningAverageMeter:
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(logpath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def train():
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    makedirs(args.save)
    logger = get_logger(os.path.join(args.save, 'training.log'))
    logger.info(f"Starting training with args: {args}")

    try:
        db_dir = download_mitdb()
    except Exception as e:
        logger.error(f"Failed to download database: {str(e)}")
        return

    all_records = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]

    train_records, test_records = train_test_split(all_records, test_size=0.2, random_state=args.seed)

    try:
        train_dataset = PhysioNetECGDataset(train_records, db_dir=db_dir)
        test_dataset = PhysioNetECGDataset(test_records, db_dir=db_dir, classes=train_dataset.classes)
    except ValueError as e:
        logger.error(f"Dataset creation failed: {str(e)}")
        return

    logger.info(f"Successfully loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    logger.info(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4)

    if args.network == 'odenet':
        feature_layers = [ODEBlock1D(ODEfunc1D(64))]
    else:
        feature_layers = [ResBlock1D(64, 64) for _ in range(3)]

    model = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=15, padding=7),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2),
        *feature_layers,
        nn.AdaptiveAvgPool1d(1),
        Flatten(),
        nn.Linear(64, len(train_dataset.classes))
    ).to(device)

    logger.info(model)
    logger.info(f"Number of parameters: {count_parameters(model)}")


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    batch_time_meter = RunningAverageMeter()
    loss_meter = RunningAverageMeter()
    accuracy_meter = RunningAverageMeter()

    best_acc = 0
    start_time = time.time()

    for epoch in range(args.nepochs):
        model.train()
        end = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = correct / data.size(0)

            loss_meter.update(loss.item())
            accuracy_meter.update(acc)
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch:03d} | Batch {batch_idx:03d} | "
                    f"Time {batch_time_meter.val:.3f} | Loss {loss_meter.val:.4f} | "
                    f"Acc {accuracy_meter.val * 100:.2f}%"
                )

        scheduler.step()

        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        test_acc = correct / len(test_loader.dataset)

        logger.info(
            f"Test Epoch {epoch:03d} | Loss {test_loss:.4f} | "
            f"Accuracy {test_acc * 100:.2f}% | Time {time.time() - start_time:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'acc': test_acc,
                'args': args
            }, os.path.join(args.save, 'best_model.pth'))

    logger.info(f"Training complete. Best accuracy: {best_acc * 100:.2f}%")


if __name__ == '__main__':
    train()