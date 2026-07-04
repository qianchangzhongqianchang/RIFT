import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

from config import device


EDGE_TYPE = ("drug", "interacts", "Protein")
REV_EDGE_TYPE = ("Protein", "interacts", "drug")


def _normalize_id(value):
    """将 123、123.0、'123' 统一为同一个字符串ID。"""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _lookup_feature(feature_dict, item_id, feature_name):
    """兼容整数、浮点字符串和普通字符串形式的特征字典键。"""
    candidates = [item_id, str(item_id), _normalize_id(item_id)]

    try:
        candidates.insert(0, int(float(item_id)))
    except (TypeError, ValueError):
        pass

    for key in candidates:
        if key in feature_dict:
            return feature_dict[key]

    raise KeyError(
        f"{feature_name} feature not found for ID {item_id!r}. "
        f"Example keys: {list(feature_dict.keys())[:5]}"
    )


def _load_protein_data_and_mapping(protein_feature_dict):
    """
    按 protein_index.csv 的行顺序构造蛋白质特征矩阵，并建立：
        relation.csv中的原始蛋白质索引 -> 图中的连续索引

    protein_index.csv:
      - 第一列：蛋白质ID
      - 第二列（若存在）：relation.csv使用的原始蛋白质索引
      - 若只有一列，则默认原始索引就是当前行号
    """
    protein_df = pd.read_csv("protein_index.csv")

    if protein_df.shape[1] < 1:
        raise ValueError("protein_index.csv must contain at least one protein ID column.")

    protein_ids = protein_df.iloc[:, 0].tolist()

    if protein_df.shape[1] >= 2:
        original_indices = pd.to_numeric(
            protein_df.iloc[:, 1],
            errors="raise",
        ).astype(np.int64).tolist()
    else:
        original_indices = list(range(len(protein_df)))

    if len(set(original_indices)) != len(original_indices):
        duplicated = pd.Series(original_indices)
        duplicated = duplicated[duplicated.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "protein_index.csv contains duplicated original indices. "
            f"Examples: {duplicated[:10]}"
        )

    protein_features = torch.stack([
        _lookup_feature(
            protein_feature_dict,
            protein_id,
            "Protein",
        )
        for protein_id in protein_ids
    ]).to(device=device, dtype=torch.float32)

    protein_original_to_new = {
        int(original_idx): new_idx
        for new_idx, original_idx in enumerate(original_indices)
    }

    return protein_features, protein_original_to_new


def _edge_list(edge_index):
    edge_index = edge_index.detach().cpu()
    return [
        (int(drug_idx), int(protein_idx))
        for drug_idx, protein_idx in edge_index.t().tolist()
    ]


def _edge_set(edge_index):
    return set(_edge_list(edge_index))


def _get_label_edge_sets(data):
    """
    返回：
      positive_edges: 正监督边集合
      negative_edges: 负监督边集合
      all_edges_list : 全部监督边列表，用于检查内部重复
    """
    store = data[EDGE_TYPE]

    edge_label_index = store.edge_label_index.detach().cpu()
    edge_label = store.edge_label.detach().cpu()

    positive_mask = edge_label > 0.5
    negative_mask = ~positive_mask

    positive_edges = _edge_set(edge_label_index[:, positive_mask])
    negative_edges = _edge_set(edge_label_index[:, negative_mask])
    all_edges_list = _edge_list(edge_label_index)

    return positive_edges, negative_edges, all_edges_list


def _freeze_eval_message_graphs(train_data, val_data, test_data):
    """
    将验证和测试对象中的消息传播图强制固定为训练消息传播图。

    这样即使主程序误用了 val_data.edge_index_dict 或
    test_data.edge_index_dict，也不会把验证正边加入消息传播。
    """
    for edge_type in (EDGE_TYPE, REV_EDGE_TYPE):
        train_edges = train_data[edge_type].edge_index
        val_data[edge_type].edge_index = train_edges.clone()
        test_data[edge_type].edge_index = train_edges.clone()


def strict_dti_split_check(
    all_positive_edge_index,
    train_data,
    val_data,
    test_data,
):
    """
    对正边划分、消息传播边、随机负采样和反向边进行严格检查。

    任一关键检查不通过时立即终止，避免带泄露的数据进入训练。
    """
    all_real_positive = _edge_set(all_positive_edge_index)

    train_mp = _edge_set(train_data[EDGE_TYPE].edge_index)
    val_mp = _edge_set(val_data[EDGE_TYPE].edge_index)
    test_mp = _edge_set(test_data[EDGE_TYPE].edge_index)

    train_pos, train_neg, train_all_list = _get_label_edge_sets(train_data)
    val_pos, val_neg, val_all_list = _get_label_edge_sets(val_data)
    test_pos, test_neg, test_all_list = _get_label_edge_sets(test_data)

    train_all = train_pos | train_neg
    val_all = val_pos | val_neg
    test_all = test_pos | test_neg

    errors = {}

    print("\n========== Strict DTI Split Check ==========")

    positive_parts = {
        "Train message positives": train_mp,
        "Train supervised positives": train_pos,
        "Validation positives": val_pos,
        "Test positives": test_pos,
    }

    positive_part_names = list(positive_parts.keys())
    for i in range(len(positive_part_names)):
        for j in range(i + 1, len(positive_part_names)):
            left_name = positive_part_names[i]
            right_name = positive_part_names[j]
            overlap = positive_parts[left_name] & positive_parts[right_name]
            name = f"{left_name} vs {right_name}"
            print(f"{name:62s}: {len(overlap)}")
            if overlap:
                errors[name] = overlap

    reconstructed_positive = train_mp | train_pos | val_pos | test_pos
    missing_positive = all_real_positive - reconstructed_positive
    unexpected_positive = reconstructed_positive - all_real_positive

    print(f"{'Missing real positives after split':62s}: {len(missing_positive)}")
    print(f"{'Unexpected positives after split':62s}: {len(unexpected_positive)}")

    if missing_positive:
        errors["Missing real positives after split"] = missing_positive
    if unexpected_positive:
        errors["Unexpected positives after split"] = unexpected_positive

    val_mp_difference = val_mp.symmetric_difference(train_mp)
    test_mp_difference = test_mp.symmetric_difference(train_mp)

    print(f"{'Validation MP differs from train MP':62s}: {len(val_mp_difference)}")
    print(f"{'Test MP differs from train MP':62s}: {len(test_mp_difference)}")

    if val_mp_difference:
        errors["Validation MP differs from train MP"] = val_mp_difference
    if test_mp_difference:
        errors["Test MP differs from train MP"] = test_mp_difference

    negative_real_positive_checks = {
        "Train negatives that are real positives": train_neg & all_real_positive,
        "Validation negatives that are real positives": val_neg & all_real_positive,
        "Test negatives that are real positives": test_neg & all_real_positive,
    }

    for name, overlap in negative_real_positive_checks.items():
        print(f"{name:62s}: {len(overlap)}")
        if overlap:
            errors[name] = overlap

    within_split_conflicts = {
        "Train positive-negative conflicts": train_pos & train_neg,
        "Validation positive-negative conflicts": val_pos & val_neg,
        "Test positive-negative conflicts": test_pos & test_neg,
    }

    for name, overlap in within_split_conflicts.items():
        print(f"{name:62s}: {len(overlap)}")
        if overlap:
            errors[name] = overlap

    cross_split_label_overlap = {
        "Train labels vs validation labels": train_all & val_all,
        "Train labels vs test labels": train_all & test_all,
        "Validation labels vs test labels": val_all & test_all,
    }

    for name, overlap in cross_split_label_overlap.items():
        print(f"{name:62s}: {len(overlap)}")
        if overlap:
            errors[name] = overlap

    duplicate_label_counts = {
        "Duplicate train label edges":
            len(train_all_list) - len(set(train_all_list)),
        "Duplicate validation label edges":
            len(val_all_list) - len(set(val_all_list)),
        "Duplicate test label edges":
            len(test_all_list) - len(set(test_all_list)),
    }

    for name, count in duplicate_label_counts.items():
        print(f"{name:62s}: {count}")
        if count > 0:
            errors[name] = count

    for split_name, data, forward_edges in (
        ("Train", train_data, train_mp),
        ("Validation", val_data, val_mp),
        ("Test", test_data, test_mp),
    ):
        reverse_edges = {
            (int(drug_idx), int(protein_idx))
            for protein_idx, drug_idx
            in data[REV_EDGE_TYPE].edge_index.detach().cpu().t().tolist()
        }

        mismatch = forward_edges.symmetric_difference(reverse_edges)
        name = f"{split_name} forward-reverse mismatch"
        print(f"{name:62s}: {len(mismatch)}")
        if mismatch:
            errors[name] = mismatch

    print("\nSample counts:")
    print(f"All unique real positives : {len(all_real_positive)}")
    print(f"Train message positives   : {len(train_mp)}")
    print(f"Train supervised positives: {len(train_pos)}")
    print(f"Train negatives           : {len(train_neg)}")
    print(f"Validation positives      : {len(val_pos)}")
    print(f"Validation negatives      : {len(val_neg)}")
    print(f"Test positives            : {len(test_pos)}")
    print(f"Test negatives            : {len(test_neg)}")
    print("============================================\n")

    if errors:
        first_items = []
        for name, value in list(errors.items())[:5]:
            if isinstance(value, set):
                examples = list(value)[:5]
                first_items.append(f"{name}: examples={examples}")
            else:
                first_items.append(f"{name}: count={value}")

        raise RuntimeError(
            "Strict DTI split/negative-sampling check failed. "
            "Do not start training.\n" + "\n".join(first_items)
        )


def get_graph(gpu_1d, protein_1d_feature, complete_3d_ids):
    """
    构建严格的DTI异构图。

    处理流程：
      1. 过滤缺少3D特征的药物；
      2. 对原始药物-蛋白质关系去重；
      3. 对药物和蛋白质原始索引建立连续映射；
      4. 映射后再次去重；
      5. RandomLinkSplit生成1:1正负样本；
      6. 固定验证/测试消息传播图为训练图；
      7. 检查正边、负边、跨集合重复和反向边。
    """
    drug_df = pd.read_csv("drug_index.csv")

    if drug_df.shape[1] < 2:
        raise ValueError(
            "drug_index.csv must contain at least two columns: "
            "drug ID and original drug index."
        )

    complete_3d_ids = {_normalize_id(x) for x in complete_3d_ids}

    drug_ids = [_normalize_id(x) for x in drug_df.iloc[:, 0].tolist()]
    drug_original_indices = pd.to_numeric(
        drug_df.iloc[:, 1],
        errors="raise",
    ).astype(np.int64).tolist()

    if len(set(drug_original_indices)) != len(drug_original_indices):
        duplicated = pd.Series(drug_original_indices)
        duplicated = duplicated[duplicated.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "drug_index.csv contains duplicated original indices. "
            f"Examples: {duplicated[:10]}"
        )

    kept_records = [
        (drug_id, original_idx)
        for drug_id, original_idx in zip(drug_ids, drug_original_indices)
        if drug_id in complete_3d_ids
    ]

    if not kept_records:
        raise ValueError(
            "No drug in drug_index.csv matches the IDs extracted from dti_encoded. "
            "Check the 3D filenames and the first column of drug_index.csv."
        )

    drug_id_map = [drug_id for drug_id, _ in kept_records]

    drug_original_to_new = {
        int(original_idx): new_idx
        for new_idx, (_, original_idx) in enumerate(kept_records)
    }

    drug_features = torch.stack([
        _lookup_feature(gpu_1d, drug_id, "1D drug")
        for drug_id in drug_id_map
    ]).to(device=device, dtype=torch.float32)

    protein_features, protein_original_to_new = _load_protein_data_and_mapping(
        protein_1d_feature
    )

    relation_df_raw = pd.read_csv("relation.csv")

    if relation_df_raw.shape[1] < 2:
        raise ValueError(
            "relation.csv must contain at least two columns: "
            "drug index and protein index."
        )

    relation_df = relation_df_raw.iloc[:, :2].copy()
    relation_df.columns = [
        "drug_original_index",
        "protein_original_index",
    ]

    relation_df["drug_original_index"] = pd.to_numeric(
        relation_df["drug_original_index"],
        errors="raise",
    ).astype(np.int64)

    relation_df["protein_original_index"] = pd.to_numeric(
        relation_df["protein_original_index"],
        errors="raise",
    ).astype(np.int64)

    raw_relation_count = len(relation_df)

    duplicate_mask = relation_df.duplicated(
        subset=[
            "drug_original_index",
            "protein_original_index",
        ],
        keep=False,
    )

    extra_duplicate_rows = int(
        relation_df.duplicated(
            subset=[
                "drug_original_index",
                "protein_original_index",
            ],
            keep="first",
        ).sum()
    )

    repeated_pair_count = int(
        relation_df.loc[
            duplicate_mask,
            [
                "drug_original_index",
                "protein_original_index",
            ],
        ].drop_duplicates().shape[0]
    )

    print("\n========== DTI Relation Check ==========")
    print(f"Raw relation rows          : {raw_relation_count}")
    print(f"Extra duplicate rows       : {extra_duplicate_rows}")
    print(f"Repeated drug-protein pairs: {repeated_pair_count}")

    if extra_duplicate_rows > 0:
        print("Examples of duplicated relations:")
        print(
            relation_df.loc[
                duplicate_mask,
                [
                    "drug_original_index",
                    "protein_original_index",
                ],
            ].head(10)
        )

    relation_df = (
        relation_df
        .drop_duplicates(
            subset=[
                "drug_original_index",
                "protein_original_index",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    unique_relation_count = len(relation_df)
    print(f"Unique relation rows       : {unique_relation_count}")
    print("========================================\n")

    filtered_relation_df = relation_df.loc[
        relation_df["drug_original_index"].isin(drug_original_to_new)
    ].copy()

    if filtered_relation_df.empty:
        raise ValueError(
            "All relations were removed after filtering drugs "
            "with complete 3D features."
        )

    unknown_protein_mask = ~filtered_relation_df[
        "protein_original_index"
    ].isin(protein_original_to_new)

    if unknown_protein_mask.any():
        unknown_indices = filtered_relation_df.loc[
            unknown_protein_mask,
            "protein_original_index",
        ].drop_duplicates().tolist()

        raise IndexError(
            "relation.csv contains protein indices that do not exist in "
            "protein_index.csv. "
            f"Examples: {unknown_indices[:10]}"
        )

    filtered_relation_df["drug_new_index"] = (
        filtered_relation_df["drug_original_index"]
        .map(drug_original_to_new)
        .astype(np.int64)
    )

    filtered_relation_df["protein_new_index"] = (
        filtered_relation_df["protein_original_index"]
        .map(protein_original_to_new)
        .astype(np.int64)
    )

    mapped_duplicate_count = int(
        filtered_relation_df.duplicated(
            subset=[
                "drug_new_index",
                "protein_new_index",
            ],
            keep="first",
        ).sum()
    )

    if mapped_duplicate_count > 0:
        print(
            f"Warning: {mapped_duplicate_count} duplicated relations "
            "appeared after index remapping and were removed."
        )

    filtered_relation_df = (
        filtered_relation_df
        .drop_duplicates(
            subset=[
                "drug_new_index",
                "protein_new_index",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    filtered_drug = filtered_relation_df[
        "drug_new_index"
    ].to_numpy(dtype=np.int64)

    filtered_protein = filtered_relation_df[
        "protein_new_index"
    ].to_numpy(dtype=np.int64)

    edge_index = torch.tensor(
        np.vstack([
            filtered_drug,
            filtered_protein,
        ]),
        dtype=torch.long,
        device=device,
    )

    num_drugs = drug_features.size(0)
    num_proteins = protein_features.size(0)
    possible_pairs = num_drugs * num_proteins
    graph_density = edge_index.size(1) / max(possible_pairs, 1)

    print("========== DTI Graph Statistics ==========")
    print(f"Drugs                    : {num_drugs}")
    print(f"Proteins                 : {num_proteins}")
    print(f"Unique positive relations: {edge_index.size(1)}")
    print(f"Possible drug-target pairs: {possible_pairs}")
    print(f"Positive graph density   : {graph_density:.8f}")
    print("==========================================\n")

    graph = HeteroData()

    graph["drug"].x = drug_features
    graph["Protein"].x = protein_features

    graph["drug"].node_idx = torch.arange(
        num_drugs,
        dtype=torch.long,
        device=device,
    )
    graph["Protein"].node_idx = torch.arange(
        num_proteins,
        dtype=torch.long,
        device=device,
    )

    graph[EDGE_TYPE].edge_index = edge_index
    graph[REV_EDGE_TYPE].edge_index = edge_index.flip(0)

    transform = RandomLinkSplit(
        num_val=0.10,
        num_test=0.10,
        is_undirected=False,
        disjoint_train_ratio=0.50,#0.3
        neg_sampling_ratio=3.0,#1
        add_negative_train_samples=True,
        edge_types=EDGE_TYPE,
        rev_edge_types=REV_EDGE_TYPE,
    )

    train_data, val_data, test_data = transform(graph)

    _freeze_eval_message_graphs(
        train_data,
        val_data,
        test_data,
    )

    strict_dti_split_check(
        edge_index,
        train_data,
        val_data,
        test_data,
    )

    print(
        f"Drugs kept: {len(drug_id_map)}/{len(drug_ids)}\n"
        f"Raw relation rows: {raw_relation_count}\n"
        f"Unique relation rows before feature filtering: {unique_relation_count}\n"
        f"Relations kept after mapping/filtering: {edge_index.size(1)}"
    )

    return train_data, val_data, test_data, drug_id_map
