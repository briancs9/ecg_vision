"""
Post-hoc probability calibration for ECG_Transformer / ECG_Model checkpoints.

Supports scalar temperature scaling, Platt scaling (affine logit transform),
and isotonic regression. By default compares all methods and saves the one
with the lowest ECE.

Usage:
  python temperature_scaling.py \\
    -m /path/to/model.pth \\
    -v /path/to/val_data \\
    -a annotations/val_annotations.csv \\
    --config config.json \\
    --method compare \\
    --output_dir calibration_outputs
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm.auto import tqdm

import config as config_module
import datasets
from inference import load_model_from_checkpoint
from utils import apply_calibration as apply_calibration_logits


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaler for binary logits."""

    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature], dtype=torch.float32))

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-4)

    def calibrated_probs(self, logits):
        return torch.sigmoid(self.forward(logits))


class PlattScaler(nn.Module):
    """Platt scaling: sigmoid(a * logit + b). Two parameters, more flexible than temperature."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, logits):
        return self.weight * logits + self.bias

    def calibrated_probs(self, logits):
        return torch.sigmoid(self.forward(logits))


class IsotonicCalibrator:
    """Non-parametric isotonic regression on uncalibrated probabilities."""

    def __init__(self, x_thresholds, y_thresholds):
        self.x_thresholds = np.asarray(x_thresholds, dtype=np.float64)
        self.y_thresholds = np.asarray(y_thresholds, dtype=np.float64)

    @classmethod
    def fit(cls, probs, labels):
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(probs.numpy(), labels.numpy())
        return cls(ir.X_thresholds_, ir.y_thresholds_)

    def calibrated_probs(self, probs):
        calibrated = np.interp(
            probs.numpy(),
            self.x_thresholds,
            self.y_thresholds,
        )
        return torch.tensor(calibrated, dtype=torch.float32)


def create_transform(config):
    """Build the same preprocessing pipeline used during training."""
    if config.transformer_model == 'transformer':
        return v2.Compose([
            v2.Lambda(lambda x: x.iloc[:, 0:8]),
            v2.Lambda(lambda x: torch.tensor(x.values, dtype=torch.float32).unsqueeze(0)),
            v2.Lambda(lambda x: datasets.column_wise_resize(x, target_rows=5000)),
            v2.Lambda(lambda x: datasets.column_wise_normalize(x)),
        ])
    return v2.Compose([
        v2.Lambda(lambda x: x.iloc[:, 0:8].values),
        v2.Lambda(lambda x: torch.tensor(x.astype(np.float32), dtype=torch.float32)),
        v2.Lambda(lambda x: x.unsqueeze(0)),
        v2.Lambda(lambda x: datasets.column_wise_resize(x, target_rows=5000)),
        v2.Lambda(lambda x: datasets.column_wise_normalize(x)),
        v2.Lambda(lambda x: x.squeeze(0)),
    ])


def create_val_loader(val_data_path, annotations_file, config, device):
    if not os.path.isdir(val_data_path):
        raise ValueError(f"val_data_path must be a directory: {val_data_path}")
    if not os.path.exists(annotations_file):
        raise ValueError(f"annotations_file does not exist: {annotations_file}")

    dataset = datasets.ECGDataset(
        annotations_file=annotations_file,
        img_dir=val_data_path,
        transform=create_transform(config),
    )
    if not dataset.has_labels:
        raise ValueError("Validation annotations must include a label column.")

    pin_memory = isinstance(device, torch.device) and device.type == 'cuda'
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return dataset, loader


@torch.no_grad()
def collect_logits_and_labels(model, data_loader):
    """Run frozen model on validation data and return raw logits + labels."""
    model.eval()
    device = next(model.parameters()).device
    all_logits = []
    all_labels = []
    all_filenames = []

    for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Collecting logits")):
        images = images.to(device, non_blocking=(device.type == 'cuda'))
        logits = model(images)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)

        all_logits.append(logits.cpu())
        all_labels.append(labels.float().cpu())

        start_idx = batch_idx * data_loader.batch_size
        end_idx = min(start_idx + len(images), len(data_loader.dataset))
        for i in range(start_idx, end_idx):
            all_filenames.append(data_loader.dataset.img_labels.iloc[i, 0])

    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
        all_filenames,
    )


def fit_temperature(logits, labels, max_iter=50, lr=0.01, device='cpu'):
    """
    Fit scalar temperature by minimizing binary cross-entropy on validation logits.

    Returns:
        Fitted TemperatureScaler and final NLL.
    """
    scaler = TemperatureScaler(init_temperature=1.0).to(device)
    _fit_nll(scaler, logits, labels, max_iter=max_iter, lr=lr, device=device)
    with torch.no_grad():
        final_nll = F.binary_cross_entropy_with_logits(scaler(logits.to(device)), labels.to(device)).item()
    return scaler, final_nll


def fit_platt(logits, labels, max_iter=50, lr=0.01, device='cpu'):
    """Fit Platt scaling (sigmoid(a*logit + b)) by minimizing NLL."""
    scaler = PlattScaler().to(device)
    _fit_nll(scaler, logits, labels, max_iter=max_iter, lr=lr, device=device)
    with torch.no_grad():
        final_nll = F.binary_cross_entropy_with_logits(scaler(logits.to(device)), labels.to(device)).item()
    return scaler, final_nll


def _fit_nll(scaler, logits, labels, max_iter=50, lr=0.01, device='cpu'):
    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=lr,
        max_iter=max_iter,
        line_search_fn='strong_wolfe',
    )
    logits = logits.to(device)
    labels = labels.to(device)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)


def fit_isotonic(logits, labels):
    """Fit isotonic regression on uncalibrated probabilities."""
    uncal_probs = torch.sigmoid(logits)
    calibrator = IsotonicCalibrator.fit(uncal_probs, labels)
    cal_probs = calibrator.calibrated_probs(uncal_probs)
    pseudo_logits = torch.logit(cal_probs.clamp(1e-6, 1 - 1e-6))
    final_nll = binary_nll(pseudo_logits, labels)
    return calibrator, final_nll


def apply_fitted_calibrator(method, calibrator, logits):
    """Return calibrated probabilities for a fitted calibrator."""
    if method == 'isotonic':
        artifact = calibration_artifact(method, calibrator)
        return torch.tensor(
            apply_calibration_logits(torch.sigmoid(logits), artifact),
            dtype=torch.float32,
        )
    calibrator.eval()
    with torch.no_grad():
        return calibrator.calibrated_probs(logits)


def calibration_artifact(method, calibrator):
    """Serialize fitted calibrator parameters for inference."""
    if method == 'temperature':
        temperature = float(calibrator.temperature.detach().cpu().item())
        return {
            'method': method,
            'temperature': temperature,
            'parameters': {'temperature': temperature},
            'calibration_rule': 'sigmoid(logit / T)',
        }
    if method == 'platt':
        weight = float(calibrator.weight.detach().cpu().item())
        bias = float(calibrator.bias.detach().cpu().item())
        return {
            'method': method,
            'parameters': {'weight': weight, 'bias': bias},
            'calibration_rule': 'sigmoid(weight * logit + bias)',
        }
    if method == 'isotonic':
        return {
            'method': method,
            'parameters': {
                'x_thresholds': calibrator.x_thresholds.tolist(),
                'y_thresholds': calibrator.y_thresholds.tolist(),
            },
            'calibration_rule': 'isotonic_regression(sigmoid(logit))',
        }
    raise ValueError(f"Unknown calibration method: {method}")


def summarize_probs(name, probs, labels, logits=None):
    """Summarize calibration metrics from probabilities (and optional logits for NLL)."""
    if logits is None:
        clipped = probs.clamp(1e-6, 1 - 1e-6)
        logits = torch.logit(clipped)
    return {
        'name': name,
        'nll': binary_nll(logits, labels),
        'brier': brier_score(probs, labels),
        'ece': expected_calibration_error(probs, labels),
    }


def binary_nll(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels).item()


def brier_score(probs, labels):
    return torch.mean((probs - labels) ** 2).item()


def expected_calibration_error(probs, labels, n_bins=15):
    """Compute ECE for binary predictions using equal-width confidence bins."""
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == 1.0:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        if not np.any(mask):
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def reliability_bins(probs, labels, n_bins=15):
    """Return bin centers, empirical accuracy, and counts for reliability plots."""
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    centers, accuracies, counts = [], [], []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == 1.0:
            mask = (probs_np >= low) & (probs_np <= high)
        else:
            mask = (probs_np >= low) & (probs_np < high)
        if not np.any(mask):
            continue
        centers.append((low + high) / 2.0)
        accuracies.append(labels_np[mask].mean())
        counts.append(mask.sum())

    return np.array(centers), np.array(accuracies), np.array(counts)


def plot_reliability_diagram(probs, labels, output_path, title=None, n_bins=20):
    centers, accuracies, counts = reliability_bins(probs, labels, n_bins=n_bins)

    label_fontsize = 14
    tick_fontsize = 12

    fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
    ax.bar(
        centers,
        accuracies,
        width=1.0 / n_bins * 0.9,
        color='#b0b0b0',
        edgecolor='#888888',
        linewidth=0.8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability', fontsize=label_fontsize)
    ax.set_ylabel('Fraction of positives', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(alpha=0.25, color='#cccccc')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Reliability diagram saved to: {output_path}")


def resolve_config_path(model_path, config_arg):
    if config_arg is not None:
        return config_arg
    model_dir = os.path.dirname(model_path)
    if not model_dir:
        return None
    config_files = [
        f for f in os.listdir(model_dir)
        if f.startswith('config_') and f.endswith('.json')
    ]
    if not config_files:
        return None
    config_files.sort(reverse=True)
    return os.path.join(model_dir, config_files[0])


def summarize_calibration(name, logits, labels, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.sigmoid(scaled_logits)
    metrics = summarize_probs(name, probs, labels, logits=scaled_logits)
    metrics['temperature'] = float(temperature)
    return metrics


def fit_calibrator(method, logits, labels, max_iter=50, lr=0.01, device='cpu'):
    if method == 'temperature':
        return fit_temperature(logits, labels, max_iter=max_iter, lr=lr, device=device)
    if method == 'platt':
        return fit_platt(logits, labels, max_iter=max_iter, lr=lr, device=device)
    if method == 'isotonic':
        return fit_isotonic(logits, labels)
    raise ValueError(f"Unknown calibration method: {method}")


def main():
    parser = argparse.ArgumentParser(
        description='Fit temperature scaling on a validation cohort for a pretrained ECG model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python temperature_scaling.py \\
    -m outputs/model_epoch_30.pth \\
    -v /data1/shaffeb1/csv_data/ \\
    -a annotations/val_annotations.csv \\
    --config config.json

  python temperature_scaling.py \\
    -m model.pth -v /path/to/val -a val.csv --output_dir calibration_run
        ''',
    )
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='Path to pretrained model checkpoint (.pth)')
    parser.add_argument('-v', '--val_data', type=str, required=True,
                        help='Path to validation data directory containing ECG CSV files')
    parser.add_argument('-a', '--annotations', type=str, required=True,
                        help='Validation annotations CSV with file and label columns')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON (auto-detected from model directory if omitted)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for temperature artifact, plots, and calibrated predictions')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override validation batch size from config')
    parser.add_argument('--max_iter', type=int, default=50,
                        help='LBFGS iterations for temperature optimization')
    parser.add_argument('--method', type=str, default='compare',
                        choices=['temperature', 'platt', 'isotonic', 'compare'],
                        help='Calibration method (default: compare all and pick best ECE)')
    parser.add_argument('--n_bins', type=int, default=20,
                        help='Number of bins for reliability diagrams (20 = 5%% bins)')
    args = parser.parse_args()

    config_json_path = resolve_config_path(args.model_path, args.config)
    if config_json_path and os.path.exists(config_json_path):
        config = config_module.Config(config_json_path=config_json_path)
        print(f"Loaded config from: {config_json_path}")
    else:
        print("Warning: No config file found, using default config")
        config = config_module.Config()

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.model_path) or '.',
        f'temperature_scaling_{timestamp}',
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {args.model_path}")
    model = load_model_from_checkpoint(args.model_path, config)
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    print(f"Loading validation data from: {args.val_data}")
    val_dataset, val_loader = create_val_loader(
        args.val_data, args.annotations, config, device,
    )
    print(f"Validation cohort size: {len(val_dataset)}")

    logits, labels, filenames = collect_logits_and_labels(model, val_loader)

    methods = ['temperature', 'platt', 'isotonic'] if args.method == 'compare' else [args.method]
    before = summarize_calibration('uncalibrated', logits, labels, temperature=1.0)
    results = {}

    for method in methods:
        print(f"\nFitting {method} calibration...")
        calibrator, fitted_nll = fit_calibrator(
            method, logits, labels, max_iter=args.max_iter, device='cpu',
        )
        cal_probs = apply_fitted_calibrator(method, calibrator, logits)
        metrics = summarize_probs(f'{method}_calibrated', cal_probs, labels)
        metrics['fitted_nll'] = fitted_nll
        results[method] = {
            'calibrator': calibrator,
            'metrics': metrics,
            'cal_probs': cal_probs,
            'artifact': calibration_artifact(method, calibrator),
        }
        print(
            f"  NLL={metrics['nll']:.4f}  "
            f"Brier={metrics['brier']:.4f}  "
            f"ECE={metrics['ece']:.4f}"
        )
        if method == 'temperature':
            print(f"  T={results[method]['artifact']['temperature']:.4f}")
        elif method == 'platt':
            p = results[method]['artifact']['parameters']
            print(f"  weight={p['weight']:.4f}  bias={p['bias']:.4f}")

    if args.method == 'compare':
        best_method = min(methods, key=lambda m: results[m]['metrics']['ece'])
        print(f"\nBest method by ECE: {best_method}")
    else:
        best_method = args.method

    best = results[best_method]
    cal_probs = best['cal_probs']
    after = best['metrics']

    print("\nCalibration summary (uncalibrated vs selected):")
    print(
        f"  {before['name']:20s}  "
        f"NLL={before['nll']:.4f}  "
        f"Brier={before['brier']:.4f}  "
        f"ECE={before['ece']:.4f}"
    )
    print(
        f"  {after['name']:20s}  "
        f"NLL={after['nll']:.4f}  "
        f"Brier={after['brier']:.4f}  "
        f"ECE={after['ece']:.4f}"
    )

    uncal_probs = torch.sigmoid(logits)

    plot_reliability_diagram(
        uncal_probs, labels,
        os.path.join(output_dir, 'reliability_uncalibrated.png'),
        title='Uncalibrated',
        n_bins=args.n_bins,
    )
    plot_reliability_diagram(
        cal_probs, labels,
        os.path.join(output_dir, 'reliability_calibrated.png'),
        title=f'{best_method} calibrated',
        n_bins=args.n_bins,
    )

    artifact = {
        **best['artifact'],
        'model_path': os.path.abspath(args.model_path),
        'val_data': os.path.abspath(args.val_data),
        'annotations': os.path.abspath(args.annotations),
        'config_json': os.path.abspath(config_json_path) if config_json_path else None,
        'validation_size': len(val_dataset),
        'metrics': {
            'uncalibrated': before,
            'calibrated': after,
            'all_methods': {m: results[m]['metrics'] for m in methods},
        },
        'fitted_at': timestamp,
    }
    artifact_path = os.path.join(output_dir, 'calibration.json')
    with open(artifact_path, 'w') as f:
        json.dump(artifact, f, indent=2)
    print(f"\nCalibration artifact saved to: {artifact_path}")
    # Backward-compatible alias for temperature-only workflows
    if best_method == 'temperature':
        with open(os.path.join(output_dir, 'temperature.json'), 'w') as f:
            json.dump(artifact, f, indent=2)

    predictions_path = os.path.join(output_dir, 'val_predictions_calibrated.csv')
    pd.DataFrame({
        'file': filenames,
        'true_label': labels.numpy(),
        'logit': logits.numpy(),
        'prob_uncalibrated': uncal_probs.numpy(),
        'prob_calibrated': cal_probs.numpy(),
    }).to_csv(predictions_path, index=False)
    print(f"Calibrated validation predictions saved to: {predictions_path}")


if __name__ == '__main__':
    main()
