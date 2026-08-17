#!/usr/bin/env python3
"""Fast evaluation of polyp segmentation masks."""

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import convolve, distance_transform_edt
from tabulate import tabulate
from tqdm import tqdm


DATASETS = ("CVC-300", "CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "ETIS-LaribPolypDB")
METRICS = (
    "mDice", "mIoU", "wFm", "Sm", "meanEm", "mae", "maxEm",
    "maxDic", "maxIoU", "meanSen", "maxSen", "meanSpe", "maxSpe",
)
THRESHOLDS = np.linspace(1.0, 0.0, 256)
EPS = np.finfo(np.float64).eps
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _object(pred, gt):
    values = pred[gt]
    x = np.mean(values)
    return 2.0 * x / (x * x + 1.0 + np.std(values) + EPS)


def _ssim(pred, gt):
    x, y = np.mean(pred), np.mean(gt)
    n = pred.size
    sigma_x2 = np.sum((pred - x) ** 2) / (n - 1 + EPS)
    sigma_y2 = np.sum((gt - y) ** 2) / (n - 1 + EPS)
    sigma_xy = np.sum((pred - x) * (gt - y)) / (n - 1 + EPS)
    alpha = 4.0 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)
    return alpha / (beta + EPS) if alpha != 0 else float(beta == 0)


def structure_measure(pred, gt):
    y = np.mean(gt)
    if y == 0:
        return 1.0 - np.mean(pred)
    if y == 1:
        return np.mean(pred)

    pred_fg = pred.copy()
    pred_fg[~gt] = 0.0
    pred_bg = 1.0 - pred
    pred_bg[gt] = 0.0
    object_score = y * _object(pred_fg, gt) + (1.0 - y) * _object(pred_bg, ~gt)

    rows, cols = np.where(gt)
    x, z = int(np.mean(rows).round()), int(np.mean(cols).round())
    slices = ((slice(None, x), slice(None, z)), (slice(x, None), slice(None, z)),
              (slice(None, x), slice(z, None)), (slice(x, None), slice(z, None)))
    region_score = sum(_ssim(pred[s], gt[s]) * gt[s].size / gt.size for s in slices)
    return max(0.0, 0.5 * (object_score + region_score))


def _gaussian_kernel(size=7, sigma=5):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


GAUSSIAN_KERNEL = _gaussian_kernel()


def weighted_fmeasure(pred, gt):
    error = np.abs(pred - gt)
    distance, nearest = distance_transform_edt(~gt, return_indices=True)
    propagated = error.copy()
    propagated[~gt] = propagated[nearest[0, ~gt], nearest[1, ~gt]]
    averaged = convolve(propagated, GAUSSIAN_KERNEL, mode="nearest")
    minimum = error.copy()
    use_average = gt & (averaged < error)
    minimum[use_average] = averaged[use_average]
    weight = np.ones_like(pred)
    weight[~gt] = 2.0 - np.exp(np.log(0.5) / 5.0 * distance[~gt])
    weighted_error = minimum * weight
    tp = np.sum(gt) - np.sum(weighted_error[gt])
    fp = np.sum(weighted_error[~gt])
    recall = 1.0 - np.mean(weighted_error[gt])
    precision = tp / (tp + fp + EPS)
    return 2.0 * recall * precision / (recall + precision + EPS)


def _threshold_metrics(pred_u8, gt):
    """Compute all 256 threshold curves from two histograms, without 256 image scans."""
    fg_hist = np.bincount(pred_u8[gt], minlength=256)
    bg_hist = np.bincount(pred_u8[~gt], minlength=256)
    fg_suffix = np.cumsum(fg_hist[::-1])[::-1]
    bg_suffix = np.cumsum(bg_hist[::-1])[::-1]
    levels = np.arange(256, dtype=np.float64) / 255.0
    indices = np.searchsorted(levels, THRESHOLDS, side="left")
    tp = fg_suffix[indices].astype(np.float64)
    fp = bg_suffix[indices].astype(np.float64)
    positives = float(gt.sum())
    negatives = float(gt.size - gt.sum())
    fn, tn = positives - tp, negatives - fp

    dice = np.zeros(256)
    iou = np.zeros(256)
    sensitivity = np.zeros(256)
    specificity = np.zeros(256)
    valid = tp > 0  # Preserve the original evaluator's TP == 0 behavior.
    dice[valid] = 2.0 * tp[valid] / (positives + tp[valid] + fp[valid])
    iou[valid] = tp[valid] / (positives + fp[valid])
    sensitivity[valid] = tp[valid] / positives
    np.divide(tn, negatives, out=specificity, where=valid & (negatives != 0))
    specificity[valid & (negatives == 0)] = np.nan

    predicted = tp + fp
    denominator = gt.size - 1 + EPS
    if positives == 0:
        enhanced = tn / denominator
    elif negatives == 0:
        enhanced = tp / denominator
    else:
        mean_gt = positives / gt.size
        mean_pred = predicted / gt.size

        def contribution(count, pred_value, gt_value):
            a = pred_value - mean_pred
            b = gt_value - mean_gt
            alignment = 2.0 * a * b / (a * a + b * b + EPS)
            return count * ((alignment + 1.0) ** 2 / 4.0)

        enhanced = (
            contribution(tp, 1.0, 1.0) + contribution(fp, 1.0, 0.0)
            + contribution(fn, 0.0, 1.0) + contribution(tn, 0.0, 0.0)
        ) / denominator
    return enhanced, dice, iou, sensitivity, specificity


def _as_u8(array, path="image"):
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        raise ValueError(f"Only 8-bit masks are supported: {path} ({array.dtype})")
    # Some inference scripts save binary masks as uint8 values {0, 1}.
    if array.size and array.max() <= 1:
        return array * 255
    return array


def _read_u8(path):
    array = np.asarray(Image.open(path))
    if array.ndim != 2:
        array = array[..., 0]
    return _as_u8(array, path)


def _evaluate_image(paths):
    pred_path, gt_path = paths
    pred_u8, gt_u8 = _read_u8(pred_path), _read_u8(gt_path)
    if pred_u8.shape != gt_u8.shape:
        raise ValueError(f"Shape mismatch: {pred_path} {pred_u8.shape} != {gt_path} {gt_u8.shape}")
    pred = pred_u8.astype(np.float64) / 255.0
    gt = gt_u8 > 127
    pred_label = pred_u8 > 127
    tp = np.sum(pred_label & gt)
    fp = np.sum(pred_label & ~gt)
    fn = np.sum(~pred_label & gt)
    tn = np.sum(~pred_label & ~gt)
    curves = _threshold_metrics(pred_u8, gt)
    return structure_measure(pred, gt), weighted_fmeasure(pred, gt), np.mean(np.abs(gt - pred)), curves, (tp, fp, fn, tn)


def _mmseg_metrics(confusion):
    """Return the foreground (polyp) row from MMseg's per-class metrics."""
    tp, fp, fn, tn = np.asarray(confusion, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = 2.0 * tp / (2.0 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return dice, iou


def _images_by_stem(folder):
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    result = {p.stem: p for p in files}
    if len(result) != len(files):
        raise ValueError(f"Duplicate image stems in {folder}")
    return result


def _pairs(pred_dir, gt_dir):
    preds, gts = _images_by_stem(pred_dir), _images_by_stem(gt_dir)
    if preds.keys() != gts.keys():
        missing_pred = sorted(gts.keys() - preds.keys())[:5]
        missing_gt = sorted(preds.keys() - gts.keys())[:5]
        raise ValueError(f"Unmatched files in {pred_dir.name}: missing predictions={missing_pred}, missing GT={missing_gt}")
    if not preds:
        raise ValueError(f"No images found in {pred_dir}")
    return [(str(preds[name]), str(gts[name])) for name in sorted(preds)]


def evaluate(pred_root, gt_root, method="TSBANet", workers=None, verbose=True):
    pred_root, gt_root = Path(pred_root), Path(gt_root)
    workers = workers or min(8, os.cpu_count() or 1)
    rows = []
    executor = None if workers == 1 else ProcessPoolExecutor(max_workers=workers)
    try:
        for dataset in tqdm(DATASETS, desc="Datasets", disable=not verbose):
            pairs = _pairs(pred_root / dataset, gt_root / dataset / "masks")
            samples = (map(_evaluate_image, pairs) if executor is None else
                       executor.map(_evaluate_image, pairs, chunksize=max(1, len(pairs) // (workers * 4))))
            samples = tqdm(samples, total=len(pairs), desc=dataset, leave=False, disable=not verbose)

            sm = wfm = mae = 0.0
            confusion = np.zeros(4, dtype=np.int64)
            curve_sums = [np.zeros(256) for _ in range(5)]
            for sample_sm, sample_wfm, sample_mae, curves, sample_confusion in samples:
                sm += sample_sm
                wfm += sample_wfm
                mae += sample_mae
                confusion += sample_confusion
                for total, curve in zip(curve_sums, curves):
                    total += curve

            count = len(pairs)
            em, dice, iou, sensitivity, specificity = (curve / count for curve in curve_sums)
            m_dice, m_iou = _mmseg_metrics(confusion)
            values = (
                m_dice, m_iou, wfm / count, sm / count, np.mean(em), mae / count,
                np.max(em), np.max(dice), np.max(iou), np.mean(sensitivity), np.max(sensitivity),
                np.mean(specificity), np.max(specificity),
            )
            rows.append([dataset, *values])
            csv_path = pred_root / f"result_mmseg_polyp_{dataset}.csv"
            new_file = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(("method", *METRICS))
                writer.writerow((method, *(f"{value:.4f}" for value in values)))
    finally:
        if executor:
            executor.shutdown()

    table = tabulate(rows, headers=("dataset", *METRICS), floatfmt=".3f")
    if verbose:
        print(table)
    return table


def _self_test():
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 256, (17, 19), dtype=np.uint8)
    gt = rng.random((17, 19)) > 0.6
    fast = _threshold_metrics(pred, gt)
    for j, threshold in enumerate(THRESHOLDS):
        binary = pred.astype(np.float64) / 255.0 >= threshold
        tp, fp = np.sum(binary & gt), np.sum(binary & ~gt)
        if tp:
            assert np.isclose(fast[1][j], 2 * tp / (gt.sum() + binary.sum()))
            assert np.isclose(fast[2][j], tp / (gt.sum() + fp))
    expected_binary = np.where(gt, 255, 0).astype(np.uint8)
    assert np.array_equal(_as_u8(gt), expected_binary)
    assert np.array_equal(_as_u8(gt.astype(np.uint8)), expected_binary)
    # TP=2, FP=1, FN=1: match class index 1 in MMseg's per-class table.
    mdice, miou = _mmseg_metrics((2, 1, 1, 6))
    assert np.isclose(mdice, 4 / 6)
    assert np.isclose(miou, 2 / 4)
    print("Self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", default="pred-image/polyp")
    parser.add_argument("--gt-root", default="TestDataset")
    parser.add_argument("--method", default="TSDBNet")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: min(8, CPU count); use 1 to disable)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        evaluate(args.pred_root, args.gt_root, args.method, args.workers, not args.quiet)


if __name__ == "__main__":
    main()
