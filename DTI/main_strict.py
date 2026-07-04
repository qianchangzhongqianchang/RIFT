import argparse
import gc
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

from config import device
from get_graph_auto_fixed_strict import get_graph
from graph_model_auto import RIFT
from utiles_auto import (
    lode1d_to_gpu,
    load_Protein_features,
    load_npy_to_gpu,
)


print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2024)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-6)
parser.add_argument("--lambda_rotation", type=float, default=0.05)
parser.add_argument("--rotation_warmup", type=int, default=50)
parser.add_argument("--val_interval", type=int, default=10)
parser.add_argument(
    "--model_path",
    type=str,
    default="model_path/Auto_best_model_ronghe_model222.pth",
)
parser.add_argument(
    "--roc_csv",
    type=str,
    default="DTI_ROC_curve.csv",
)
parser.add_argument(
    "--roc_png",
    type=str,
    default="DTI_ROC_curve.png",
)
args = parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_drug_2d_features2(drug_ids, gpu_2d):
    tensors = []

    for drug_id in drug_ids:
        key = int(float(drug_id))
        feature = gpu_2d.get(key)

        if feature is None:
            raise KeyError(
                f"找不到药物 {drug_id} 的二维特征，"
                f"gpu_2d中的部分键为：{list(gpu_2d.keys())[:10]}"
            )

        tensors.append(feature)

    return torch.stack(tensors, dim=0)


def get_drug_features3d(drug_ids, drug_features_dict):
    tensors = []

    for drug_id in drug_ids:
        normalized_id = str(int(float(drug_id)))

        candidate_names = [
            f"{normalized_id}_output_encoded.npy",
            f"{normalized_id}_encoded.npy",
        ]

        feature = None

        for file_name in candidate_names:
            if file_name in drug_features_dict:
                feature = drug_features_dict[file_name]
                break

        if feature is None:
            raise KeyError(
                f"找不到药物 {normalized_id} 的三维特征，"
                f"尝试过：{candidate_names}"
            )

        tensors.append(feature)

    return torch.stack(tensors, dim=0)


def _copy_x_dict(data):
    """
    模型内部目前会给 x_dict['drug'] 重新赋值。
    传入浅拷贝，避免修改 train_data 中长期保存的节点特征字典。
    """
    return {
        node_type: feature
        for node_type, feature in data.x_dict.items()
    }


@torch.no_grad()
def evaluate(
    model,
    train_data,
    label_data,
    drug_2d_features,
    drug_3d_features,
):
    """
    严格评估：
      - 节点表示始终只由 train_data.edge_index_dict 计算；
      - 待预测边和标签来自 label_data。
    """
    model.eval()

    outputs = model(
        _copy_x_dict(train_data),
        train_data.edge_index_dict,
        drug_2d_features,
        drug_3d_features,
    )

    result = model.compute_loss(
        outputs,
        label_data,
        lambda_rotation=0.0,
    )

    (
        auc_value,
        aupr_value,
        accuracy_value,
        precision_value,
        recall_value,
        f1_value,
    ) = model.test(
        result["scores"],
        result["labels"],
    )

    metrics = {
        "loss": float(result["loss"].item()),
        "auc": float(auc_value),
        "aupr": float(aupr_value),
        "accuracy": float(accuracy_value),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "f1": float(f1_value),
    }

    return metrics, result["scores"], result["labels"]


def save_validation_roc(scores, labels, auc_value):
    probabilities = torch.sigmoid(scores).detach().cpu().numpy().reshape(-1)
    targets = labels.detach().cpu().numpy().reshape(-1)

    fpr, tpr, thresholds = roc_curve(
        targets,
        probabilities,
    )

    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }).to_csv(args.roc_csv, index=False)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_value:.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("DTI Validation ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.roc_png, dpi=300)
    plt.close()


def main():
    set_random_seed(args.seed)

    npy_files_dir = "dti_encoded"

    npy_files = [
        os.path.join(npy_files_dir, file_name)
        for file_name in os.listdir(npy_files_dir)
        if file_name.endswith(".npy")
    ]

    gpu_data_3d = load_npy_to_gpu(
        npy_files,
        device,
    )

    gpu_1d = lode1d_to_gpu(
        "drug_1d_fingerprints.csv",
        device=device,
    )

    gpu_2d = lode1d_to_gpu(
        "drug_2d.csv",
        device=device,
    )

    protein_1d_feature = load_Protein_features(
        "output_esm2.csv",
        device=device,
    )

    complete_3d_ids = {
        file_name
        .replace("_output_encoded.npy", "")
        .replace("_encoded.npy", "")
        for file_name in gpu_data_3d.keys()
    }

    train_data, val_data, test_data, drug_id_map = get_graph(
        gpu_1d,
        protein_1d_feature,
        complete_3d_ids,
    )

    # 全部划分包含相同节点集合，因此药物属性只需堆叠一次。
    drug_2d_features = get_drug_2d_features2(
        drug_id_map,
        gpu_2d,
    ).to(device)

    drug_3d_features = get_drug_features3d(
        drug_id_map,
        gpu_data_3d,
    ).to(device)

    model = RIFT().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    os.makedirs(
        os.path.dirname(args.model_path) or ".",
        exist_ok=True,
    )

    best_auc = -float("inf")
    best_epoch = 0

    progress = tqdm(
        range(args.epochs),
        desc="Training",
        dynamic_ncols=True,
    )

    for epoch_index in progress:
        epoch = epoch_index + 1

        model.train()
        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            _copy_x_dict(train_data),
            train_data.edge_index_dict,
            drug_2d_features,
            drug_3d_features,
        )

        rotation_weight = (
            args.lambda_rotation
            * min(
                1.0,
                epoch / max(args.rotation_warmup, 1),
            )
        )

        loss_dict = model.compute_loss(
            outputs,
            train_data,
            lambda_rotation=rotation_weight,
        )

        loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at epoch {epoch}: "
                f"{loss.item()}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm at epoch {epoch}: "
                f"{grad_norm.item()}"
            )

        optimizer.step()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            rotation=f"{loss_dict['rotation_loss'].item():.4f}",
        )

        if epoch % args.val_interval == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics, val_scores, val_labels = evaluate(
                model,
                train_data,
                val_data,
                drug_2d_features,
                drug_3d_features,
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

                torch.save(
                    state_dict,
                    args.model_path,
                )

                save_validation_roc(
                    val_scores,
                    val_labels,
                    val_metrics["auc"],
                )

                progress.write(
                    f"Best model saved: epoch={best_epoch}, "
                    f"AUC={best_auc:.4f}, "
                    f"path={args.model_path}"
                )

    best_state = torch.load(
        args.model_path,
        map_location=device,
    )

    model.load_state_dict(best_state)

    test_metrics, _, _ = evaluate(
        model,
        train_data,
        test_data,
        drug_2d_features,
        drug_3d_features,
    )

    print("\n========== Strict Final Test Results ==========")
    print(f"Best Epoch : {best_epoch}")
    print(f"Test Loss  : {test_metrics['loss']:.4f}")
    print(f"Test AUC   : {test_metrics['auc']:.4f}")
    print(f"Test AUPR  : {test_metrics['aupr']:.4f}")
    print(f"Test ACC   : {test_metrics['accuracy']:.4f}")
    print(f"Test Prec  : {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1    : {test_metrics['f1']:.4f}")
    print("===============================================\n")

    del gpu_data_3d
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
