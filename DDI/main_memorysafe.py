import argparse
import gc
import hashlib
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
#from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from tqdm import tqdm

from config import device
from get_graph import get_graph
from graph_model_memorysafe import RIFTMolGNN_DDI


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-6)
parser.add_argument("--lambda_rotation", type=float, default=0.05)
parser.add_argument("--rotation_warmup", type=int, default=50)
parser.add_argument("--val_interval", type=int, default=10)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument(
    "--amp_dtype",
    choices=["auto", "bf16", "fp16", "fp32"],
    default="auto",
    help="auto优先BF16；不支持BF16时使用FP32，避免3D特征转FP16溢出。",
)
parser.add_argument(
    "--rotation_batch_size",
    type=int,
    default=256,
    help="旋转一致性分支每轮抽样的药物数；0表示全部药物。",
)
parser.add_argument(
    "--no_saved_tensor_offload",
    action="store_true",
    help="关闭反向传播保存张量的CPU卸载。默认开启以节省显存。",
)

parser.add_argument("--feature_3d_dir", type=str, default="ddi_encoded")
parser.add_argument("--feature_1d_file", type=str, default="drug_1d_fingerprints.csv")
parser.add_argument("--feature_2d_file", type=str, default="drug_2d.csv")
parser.add_argument("--feature_3d_cache", type=str, default="ddi_3d_cache_fp32.npy")
parser.add_argument("--rebuild_3d_cache", action="store_true")

parser.add_argument(
    "--model_path",
    type=str,
    default="model_path/DDI_fullgraph_best_model.pth",
)
parser.add_argument(
    "--prediction_file",
    type=str,
    default="predictions/DDI_fullgraph_test_predictions.csv",
)
args = parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_name(value):
    """Make CSV IDs and graph drug names comparable."""
    text = str(value).strip()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def load_csv_feature_matrix(file_path, idx_to_node, feature_name):
    """Load one feature row per graph node, ordered by graph node index."""
    df = pd.read_csv(file_path, header=None)
    names = [normalize_name(value) for value in df.iloc[:, 0].tolist()]

    values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        bad_rows = values.index[values.isna().any(axis=1)].tolist()[:10]
        raise ValueError(
            f"{feature_name} contains NaN/non-numeric values; rows={bad_rows}"
        )

    matrix = values.to_numpy(dtype=np.float32)
    row_by_name = {name: index for index, name in enumerate(names)}
    ordered_names = [
        normalize_name(idx_to_node[index])
        for index in range(len(idx_to_node))
    ]

    missing = [name for name in ordered_names if name not in row_by_name]
    if missing:
        raise KeyError(
            f"{feature_name} is missing {len(missing)} graph drugs; "
            f"examples={missing[:10]}"
        )

    order = np.asarray([row_by_name[name] for name in ordered_names], dtype=np.int64)
    tensor = torch.from_numpy(np.ascontiguousarray(matrix[order]))
    print(f"{feature_name} loaded: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    return tensor


def node_signature(idx_to_node):
    names = [normalize_name(idx_to_node[index]) for index in range(len(idx_to_node))]
    signature = hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()
    return names, signature


def find_3d_file(feature_dir, drug_name):
    candidates = [
        os.path.join(feature_dir, f"{drug_name}_output_encoded.npy"),
        os.path.join(feature_dir, f"{drug_name}_encoded.npy"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No 3D feature found for {drug_name}; tried: {candidates}"
    )


def load_or_build_3d_cache(idx_to_node, feature_dir, cache_path, rebuild=False):
    """Build/load one contiguous FP32 tensor ordered by graph node index."""
    names, signature = node_signature(idx_to_node)
    cache_path = os.path.abspath(cache_path)
    meta_path = cache_path + ".json"

    cache_valid = False
    if not rebuild and os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as file:
                meta = json.load(file)
            cache_valid = (
                meta.get("node_signature") == signature
                and meta.get("node_count") == len(names)
                and meta.get("dtype") == "float32"
            )
        except (OSError, ValueError, json.JSONDecodeError):
            cache_valid = False

    if not cache_valid:
        print("Building FP32 3D cache...")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

        first_path = find_3d_file(feature_dir, names[0])
        first = np.load(first_path, mmap_mode="r")
        feature_shape = tuple(first.shape)

        temporary_path = cache_path + ".building.npy"
        cache = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(names), *feature_shape),
        )

        for index, drug_name in enumerate(tqdm(names, desc="Building 3D cache")):
            path = find_3d_file(feature_dir, drug_name)
            feature = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)

            if tuple(feature.shape) != feature_shape:
                raise ValueError(
                    f"3D shape mismatch for {drug_name}: "
                    f"expected={feature_shape}, actual={tuple(feature.shape)}"
                )
            if not np.isfinite(feature).all():
                raise ValueError(f"3D feature contains NaN/Inf: {path}")

            cache[index] = feature

        cache.flush()
        del cache
        os.replace(temporary_path, cache_path)

        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "node_signature": signature,
                    "node_count": len(names),
                    "dtype": "float32",
                    "feature_shape": list(feature_shape),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print(f"3D cache created: {cache_path}")
    else:
        print(f"Using existing 3D cache: {cache_path}")

    array = np.load(cache_path, mmap_mode=None)
    if not np.isfinite(array).all():
        raise ValueError(
            f"3D cache contains NaN/Inf: {cache_path}. Delete and rebuild it."
        )

    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
    size_gib = tensor.numel() * tensor.element_size() / 1024**3
    print(
        f"3D loaded: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"size={size_gib:.2f} GiB"
    )
    return tensor


def calculate_metrics(scores, labels, threshold):
    probabilities = torch.sigmoid(scores).detach().float().cpu().numpy().reshape(-1)
    targets = labels.detach().float().cpu().numpy().reshape(-1)

    if np.unique(targets).size < 2:
        raise ValueError("ROC-AUC requires both positive and negative labels.")

    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "auc": roc_auc_score(targets, probabilities),
        "aupr": average_precision_score(targets, probabilities),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
    }, probabilities, targets


def resolve_precision():
    if device.type != "cuda":
        return False, torch.float32

    if args.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前GPU不支持BF16，请使用 --amp_dtype fp32。")
        return True, torch.bfloat16

    if args.amp_dtype == "fp16":
        return True, torch.float16

    if args.amp_dtype == "fp32":
        return False, torch.float32

    # auto：优先BF16。旧GPU回退FP32，避免曾经出现的FP16溢出。
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return False, torch.float32


def autocast_context(enabled, dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate(
    model,
    data,
    feature_1d,
    feature_2d,
    feature_3d,
    threshold,
    amp_enabled,
    amp_dtype,
):
    model.eval()

    with autocast_context(amp_enabled, amp_dtype):
        outputs = model(
            data,
            feature_1d,
            feature_2d,
            feature_3d,
            compute_rotation=False,
            return_attention=False,
        )
        loss_dict = model.compute_loss(outputs, data, lambda_rotation=0.0)
    metrics, probabilities, targets = calculate_metrics(
        loss_dict["scores"],
        loss_dict["labels"],
        threshold,
    )
    metrics["loss"] = float(loss_dict["loss"].item())
    fpr, tpr, thresholds = roc_curve(targets, probabilities)
    return metrics, probabilities, targets, fpr, tpr, thresholds


def print_metrics(prefix, epoch, metrics):
    print(
        f"{prefix} {epoch:4d} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | "
        f"AUPR: {metrics['aupr']:.4f} | "
        f"ACC: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )


def move_full_features_to_device(
    feature_1d,
    feature_2d,
    feature_3d,
    amp_enabled,
    amp_dtype,
):
    total_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (feature_1d, feature_2d, feature_3d)
    )
    print(f"Full FP32 input feature memory: {total_bytes / 1024**3:.2f} GiB")

    # 1D/2D很小，保持FP32。3D在BF16/FP16模式下按半精度常驻GPU。
    feature_3d_dtype = amp_dtype if amp_enabled else torch.float32

    try:
        feature_1d = feature_1d.to(device=device, dtype=torch.float32)
        feature_2d = feature_2d.to(device=device, dtype=torch.float32)
        feature_3d = feature_3d.to(device=device, dtype=feature_3d_dtype)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "输入特征移动到GPU时显存不足。优先使用支持BF16的GPU，"
            "或减小 --rotation_batch_size；不要先改图拆分和预测损失。"
        ) from error

    actual_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (feature_1d, feature_2d, feature_3d)
    )
    print(
        f"Actual GPU input memory: {actual_bytes / 1024**3:.2f} GiB, "
        f"3D dtype={feature_3d.dtype}"
    )
    return feature_1d, feature_2d, feature_3d


def main():
    print(f"Using device: {device}")
    set_random_seed(args.seed)

    amp_enabled, amp_dtype = resolve_precision()
    print(f"AMP enabled: {amp_enabled}, compute dtype: {amp_dtype}")

    train_data, val_data, test_data, idx_to_node, _ = get_graph()

    feature_1d_cpu = load_csv_feature_matrix(
        args.feature_1d_file,
        idx_to_node,
        "1D",
    )
    feature_2d_cpu = load_csv_feature_matrix(
        args.feature_2d_file,
        idx_to_node,
        "2D",
    )
    feature_3d_cpu = load_or_build_3d_cache(
        idx_to_node,
        args.feature_3d_dir,
        args.feature_3d_cache,
        rebuild=args.rebuild_3d_cache,
    )

    feature_1d, feature_2d, feature_3d = move_full_features_to_device(
        feature_1d_cpu,
        feature_2d_cpu,
        feature_3d_cpu,
        amp_enabled,
        amp_dtype,
    )
    del feature_1d_cpu, feature_2d_cpu, feature_3d_cpu
    gc.collect()

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    model = RIFTMolGNN_DDI(
        rotation_batch_size=args.rotation_batch_size,
    ).to(device)
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.prediction_file) or ".", exist_ok=True)

    best_auc = -float("inf")
    best_epoch = 0

    progress = tqdm(range(args.epochs), desc="Full-graph training", dynamic_ncols=True)
    for epoch_index in progress:
        epoch = epoch_index + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)

        rotation_weight = args.lambda_rotation * min(
            1.0,
            epoch / max(args.rotation_warmup, 1),
        )

        offload_context = (
            nullcontext()
            if args.no_saved_tensor_offload
            else torch.autograd.graph.save_on_cpu(pin_memory=True)
        )

        # 保持完整全图前向、完整预测损失和原旋转正则；只改变精度与激活存放位置。
        with offload_context:
            with autocast_context(amp_enabled, amp_dtype):
                outputs = model(
                    train_data,
                    feature_1d,
                    feature_2d,
                    feature_3d,
                    compute_rotation=True,
                    return_attention=False,
                )
                loss_dict = model.compute_loss(
                    outputs,
                    train_data,
                    lambda_rotation=rotation_weight,
                )
                loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {epoch}: {loss.item()}"
            )

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at epoch {epoch}: {grad_norm.item()}"
            )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            rotation=f"{loss_dict['rotation_loss'].item():.4f}",
        )

        if epoch_index % args.val_interval == 0 or epoch == args.epochs:
            val_metrics, val_prob, val_label, fpr, tpr, thresholds = evaluate(
                model,
                val_data,
                feature_1d,
                feature_2d,
                feature_3d,
                args.threshold,
                amp_enabled,
                amp_dtype,
            )
            progress.write(
                f"Epoch {epoch:4d} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"AUPR: {val_metrics['aupr']:.4f} | "
                f"ACC: {val_metrics['accuracy']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"F1: {val_metrics['f1']:.4f}"
            )

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_epoch = epoch
                state_dict = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
                torch.save(state_dict, args.model_path)
                progress.write(
                    f"Best model saved: epoch={best_epoch}, "
                    f"AUC={best_auc:.4f}, path={args.model_path}"
                )
                pd.DataFrame({
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds
                }).to_csv("DDI_ROC_curve.csv", index=False)
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={val_metrics['auc']:.4f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC Curve")
                plt.legend()
                plt.savefig("DDI_ROC_curve.png")
                plt.close()

    best_state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(best_state)

    test_metrics, test_probabilities, test_targets = evaluate(
        model,
        test_data,
        feature_1d,
        feature_2d,
        feature_3d,
        args.threshold,
        amp_enabled,
        amp_dtype,
    )

    print("\n========== Final Test Results ==========")
    print(f"Best Epoch : {best_epoch}")
    print(f"Test Loss  : {test_metrics['loss']:.4f}")
    print(f"Test AUC   : {test_metrics['auc']:.4f}")
    print(f"Test AUPR  : {test_metrics['aupr']:.4f}")
    print(f"Test ACC   : {test_metrics['accuracy']:.4f}")
    print(f"Test Prec  : {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1    : {test_metrics['f1']:.4f}")
    print("========================================")

    pd.DataFrame(
        {
            "probability": test_probabilities,
            "label": test_targets,
        }
    ).to_csv(args.prediction_file, index=False)
    print(f"Test predictions saved to: {args.prediction_file}")


if __name__ == "__main__":
    main()
