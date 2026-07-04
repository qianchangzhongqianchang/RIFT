import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
from config import device



class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()
        # 第一层卷积
        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)
        # 第二层卷积
        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)
        # 第三层卷积
        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
        self.nor = nn.BatchNorm1d(256)
        

    def forward(self, x):
        x = x.unsqueeze(1)  
        

        x = F.relu(self.conv1(x))  
        x = self.pool(x)  

        
        x = F.relu(self.conv2(x))  
        x = self.pool(x)  
        
        x = F.relu(self.conv3(x))  
        x = self.pool(x)  


        x = x.mean(dim=[1, 2])
        x = x.unsqueeze(1)
        x = self.pool2(x)
        x = x.squeeze(1)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 64, 1, 64, 64)
            nn.ReLU(True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 64, 1, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),  # (32, 3, 8, 512, 512)
            nn.Sigmoid()  # For normalized pixel values between [0, 1]
        )
    def forward(self, x):
        x = self.encoder(x)

        encode = self.bottleneck(x)

        x = self.decoder(encode)

        return encode,x


class gate(nn.Module):
    def __init__(self):
        super(gate, self).__init__()
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):

        x1 = self.lin1(x1)
        x2 = self.lin1(x2)
        x3 = self.lin1(x3)
        x1 = self.lin2(x1)
        x2 = self.lin2(x2)
        x3 = self.lin2(x3)
        x1 = self.lin3(x1)
        x2 = self.lin3(x2)
        x3 = self.lin3(x3)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        x1 = self.sigmoid(x1)
        x2 = self.sigmoid(x2)
        x3 = self.sigmoid(x3)
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(0)
        if x3.dim() == 1:
            x3 = x3.unsqueeze(0)
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.softmax(x)
        return x

class aaaa(nn.Module):
    def __init__(self,
                 init_bias_c: float = 1.0,
                 max_epoch: int = 300,
                 max_bias: float = 5.0):
        super(aaaa, self).__init__()
        # 1) 三路输入共享的 BatchNorm
        # self.bn_input = nn.BatchNorm1d(256)
        self.bn_input = nn.BatchNorm1d(256)

        # 2) 三层 MLP（shared）
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 32)

        # 3) 池化 + softmax
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)

        # 4) 可训练静态偏置
        self.static_bias = nn.Parameter(
            torch.tensor([0., 0., init_bias_c], dtype=torch.float)
        )

        # 5) 动态偏置相关超参
        self.max_epoch = max_epoch
        self.max_bias  = max_bias

    def forward(self, x1, x2, x3, epoch: int = None):
        # --- 统一归一化到相同分布 ---
        x1 = self.bn_input(x1)
        x2 = self.bn_input(x2)

        x3 = self.bn_input(x3)

        # --- MLP 分支 ---
        def branch(x):
            x = F.relu(self.lin1(x))
            x = F.relu(self.lin2(x))
            x = F.relu(self.lin3(x))
            return self.pool(x.unsqueeze(1)).squeeze(1)  # -> [B, 32]

        a = branch(x1)
        b = branch(x2)
        c = branch(x3)

        # --- 拼接 logits ---
        logits = torch.cat([a, b, c], dim=1)  # [B,3]

        # --- 计算动态偏置 ---
        if epoch is not None:
            bias_c = (epoch / float(self.max_epoch)) * self.max_bias
        else:
            bias_c = 0.0
        dynamic_bias = torch.tensor([0., 0., bias_c],
                                    device=logits.device,
                                    dtype=logits.dtype)

        # --- 应用静态 + 动态偏置，再做 softmax ---
        logits = logits + self.static_bias + dynamic_bias
        weights = self.softmax(logits)         # [B,3]

        return weights












class conv2d(nn.Module):
    def __init__(self):
        super(conv2d, self).__init__()
        self.conv1 = nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = self.conv3(x)

        x = self.pool3(x)


        return x





class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3 ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(32)
        self.nor2 = torch.nn.BatchNorm1d(16)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)
        
        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)
        x = self.nor(x)
        x = self.relu(x)
        #x = self.drop(x)
        x2 = self.l2(x)
        x = self.l2(x)
        #x = self.drop(x)
        x = self.nor2(x)
        x = self.relu(x) 
        x = self.l3(x)


        return x,x2
    

# class Directional3DProcessor(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Directional3DProcessor, self).__init__()
#
#         self.conv_fr = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_bb = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.conv_tl = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, encoded_3d):  # [B, 128, 6, 32, 32]
#         fr = encoded_3d[:, :, 0:2]  # [B, 128, 2, 32, 32]
#         bb = encoded_3d[:, :, 2:4]
#         tl = encoded_3d[:, :, 4:6]
#
#         fr_out = self.conv_fr(fr)
#         bb_out = self.conv_bb(bb)
#         tl_out = self.conv_tl(tl)
#
#
#         combined = torch.cat([fr_out, bb_out, tl_out], dim=2)
#         return combined
class SixViewRelationEncoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.view_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.view_embedding = nn.Parameter(
            torch.randn(1, 6, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

        self.query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.pool = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, return_attention=False):
        """
        x: [N, 128, 6, H, W]

        return_attention=False 时不计算注意力权重，训练更快、显存更省。
        """

        if x.dim() != 5 or x.size(2) != 6:
            raise ValueError(
                f"Expected [N,C,6,H,W], got {x.shape}"
            )

        n, c, v, h, w = x.shape

        # 每个视角单独编码
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(n * v, c, h, w)

        tokens = self.view_encoder(x).flatten(1)
        tokens = tokens.reshape(n, v, -1)

        tokens = tokens + self.view_embedding
        tokens = self.transformer(tokens)

        query = self.query.expand(n, -1, -1)

        pooled, attention = self.pool(
            query,
            tokens,
            tokens,
            need_weights=return_attention,
        )

        feature = self.norm(
            pooled.squeeze(1)
        )

        if attention is not None:
            attention = attention.squeeze(1)

        return feature, attention

class InteractionConditionedFusion(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        dropout=0.1,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z1,
        z2,
        z3,
        condition,
    ):
        """
        z1,z2,z3: [E,128]
        condition: [E,128]
        """

        modalities = torch.stack(
            [z1, z2, z3],
            dim=1,
        )  # [E,3,128]

        condition = condition.unsqueeze(1).expand(
            -1,
            3,
            -1,
        )

        pair_features = torch.cat(
            [
                modalities,
                condition,
                modalities * condition,
                torch.abs(modalities - condition),
            ],
            dim=-1,
        )

        logits = self.scorer(
            pair_features
        ).squeeze(-1)

        weights = torch.softmax(
            logits,
            dim=1,
        )

        fused = (
            modalities * weights.unsqueeze(-1)
        ).sum(dim=1)

        return fused, weights

ROTATION_PERMUTATIONS = torch.tensor(
    [
        [0, 1, 2, 3, 4, 5],
        [2, 3, 1, 0, 4, 5],
        [1, 0, 3, 2, 4, 5],
        [3, 2, 0, 1, 4, 5],
    ],
    dtype=torch.long,
)

class RIFTMolGNN_DDI(nn.Module):
    def __init__(self, rotation_batch_size=256):
        super().__init__()
        # 仅对旋转一致性正则项抽样；主预测分支仍使用全部药物。
        # 该抽样均值是完整旋转损失均值的无偏估计，可显著减少第二次3D编码的显存。
        self.rotation_batch_size = rotation_batch_size

        self.register_buffer(
            "rotation_permutations",
            ROTATION_PERMUTATIONS,
        )

        self.view_encoder = SixViewRelationEncoder(
            in_channels=128,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

        self.project_1d = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.project_2d = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.project_3d = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.graph_input = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)

        self.norm1 = nn.BatchNorm1d(128)
        self.norm2 = nn.BatchNorm1d(64)
        self.norm3 = nn.BatchNorm1d(32)

        self.activation = nn.LeakyReLU(0.2)

        self.context_projector = nn.Linear(
            32,
            128,
        )

        self.conditioned_fusion = (
            InteractionConditionedFusion(
                hidden_dim=128,
                dropout=0.1,
            )
        )

        # 5个128维对称关系向量
        self.edge_predictor = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(
            self,
            batch,
            drug_1d_features,
            drug_2d_features,
            drug_3d_features,
            compute_rotation=True,
            return_attention=False,
    ):
        # 统一为128维
        x1 = F.adaptive_avg_pool1d(
            drug_1d_features,
            128,
        )

        x2 = F.adaptive_avg_pool1d(
            drug_2d_features,
            128,
        )

        z1 = self.project_1d(x1)
        z2 = self.project_2d(x2)

        z3_raw, view_attention = self.view_encoder(
            drug_3d_features,
            return_attention=return_attention,
        )
        z3 = self.project_3d(z3_raw)

        if self.training and compute_rotation:
            num_drugs = drug_3d_features.size(0)
            rotation_batch_size = self.rotation_batch_size

            if rotation_batch_size is not None and rotation_batch_size > 0:
                rotation_batch_size = min(rotation_batch_size, num_drugs)
                rotation_index = torch.randperm(
                    num_drugs,
                    device=drug_3d_features.device,
                )[:rotation_batch_size]
                rotation_input = drug_3d_features[rotation_index]
                z3_reference = z3[rotation_index]
            else:
                rotation_input = drug_3d_features
                z3_reference = z3

            rotated_views = self.rotate_views(rotation_input)

            z3_rot_raw, _ = self.view_encoder(
                rotated_views,
                return_attention=False,
            )
            z3_rot = self.project_3d(z3_rot_raw)

            rotation_loss = (
                1.0
                - F.cosine_similarity(
                    z3_reference,
                    z3_rot,
                    dim=-1,
                ).mean()
            )
        else:
            rotation_loss = z3.new_tensor(0.0)

        node_feature = self.graph_input(
            torch.cat(
                [z1, z2, z3],
                dim=-1,
            )
        )

        edge_index = batch.edge_index.to(
            node_feature.device
        )

        graph_feature = self.conv1(
            node_feature,
            edge_index,
        )
        graph_feature = self.activation(
            self.norm1(graph_feature)
        )

        graph_feature = self.conv2(
            graph_feature,
            edge_index,
        )
        graph_feature = self.activation(
            self.norm2(graph_feature)
        )

        graph_feature = self.conv3(
            graph_feature,
            edge_index,
        )
        graph_feature = self.activation(
            self.norm3(graph_feature)
        )

        return {
            "z1": z1,
            "z2": z2,
            "z3": z3,
            "graph_feature": graph_feature,
            "rotation_loss": rotation_loss,
            "view_attention": view_attention,
        }

    def compute_loss(
            self,
            outputs,
            batch,
            lambda_rotation=0.05,
    ):
        edge_index = batch.edge_label_index
        labels = batch.edge_label.float().to(
            edge_index.device
        )

        scores, modality_weights = self.decode_edges(
            outputs,
            edge_index,
        )

        prediction_loss = (
            F.binary_cross_entropy_with_logits(
                scores,
                labels,
            )
        )

        rotation_loss = outputs["rotation_loss"]

        total_loss = (
                prediction_loss
                + lambda_rotation * rotation_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss,
            "rotation_loss": rotation_loss,
            "scores": scores,
            "labels": labels,
            "modality_weights": modality_weights,
        }

    def rotate_views(self, x):
        n, c, v, h, w = x.shape

        rotation_ids = torch.randint(
            low=0,
            high=self.rotation_permutations.size(0),
            size=(n,),
            device=x.device,
        )

        permutations = self.rotation_permutations[
            rotation_ids
        ]

        x = x.permute(0, 2, 1, 3, 4)

        index = permutations[
                :, :, None, None, None
                ].expand(-1, -1, c, h, w)

        x = torch.gather(
            x,
            dim=1,
            index=index,
        )

        return x.permute(
            0, 2, 1, 3, 4
        ).contiguous()

    def decode_edges(self,outputs,edge_index,):
        src = edge_index[0]
        dst = edge_index[1]

        graph_context = self.context_projector(
            outputs["graph_feature"]
        )

        context_i = graph_context[src]
        context_j = graph_context[dst]

        # 药物i根据药物j选择自身模态
        drug_i, weights_i = self.conditioned_fusion(
            outputs["z1"][src],
            outputs["z2"][src],
            outputs["z3"][src],
            context_j,
        )

        # 药物j根据药物i选择自身模态
        drug_j, weights_j = self.conditioned_fusion(
            outputs["z1"][dst],
            outputs["z2"][dst],
            outputs["z3"][dst],
            context_i,
        )

        # 对称边表示
        pair_feature = torch.cat(
            [
                drug_i + drug_j,
                drug_i * drug_j,
                torch.abs(drug_i - drug_j),
                context_i + context_j,
                context_i * context_j,
            ],
            dim=-1,
        )

        scores = self.edge_predictor(
            pair_feature
        ).squeeze(-1)

        modality_weights = torch.stack(
            [weights_i, weights_j],
            dim=1,
        )  # [E,2,3]

        return scores, modality_weights

class MolVisGNN1(nn.Module):
    def __init__(self,):
        super(MolVisGNN1, self).__init__()

        self.conv1 = SAGEConv(128, 128)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.sp = Directional3DProcessor(128,64)
        self.con = conv2d()
        self.lne = nn.Linear(768,128)
        self.mlp_pre = mlp_pre(64,32,16,1)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.gate = gate()
        self.aaa = aaaa()
        self.intermediate_feature = None
        self.intermediate_gradient = None
    def forward(self, batch,drug_1d_features,drug_2d_features,drug_3d_features):

        encode_3d = drug_3d_features
        encode_3d = self.sp(encode_3d)
        encode_3d = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)



        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256)

        weights = self.aaa(drug_1d_features,drug_2d_features, encode_3d)
        # weights = self.gate(drug_1d_features, drug_2d_features, encode_3d)
        self.saved_x = weights
        w1 = weights[:,0].unsqueeze(1)
        w2 = weights[:,1].unsqueeze(1)
        w3 = weights[:,2].unsqueeze(1)
        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3

        drug_feature = torch.cat((drug_1d_features,drug_2d_features,drug_3d_features), dim=1)

        drug_feature = self.lne(drug_feature)

        drug_feature = self.nor(drug_feature)
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature

        edge_index = batch.edge_index.to(device)

        x_dict = self.conv1(drug_feature, edge_index)
        x_dict= self.nor(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv2(x_dict, edge_index)
        x_dict = self.nor2(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv3(x_dict,  edge_index)
        x_dict = self.nor3(x_dict)
        x_dict = self.relu(x_dict)


        return x_dict,self.saved_x


    def compute_loss(self, out, batch):
        edge_index = batch.edge_label_index
        labels = batch.edge_label


        src = edge_index[0]
        dst = edge_index[1]

        drug1 = out[src]  #
        drug2 = out[dst]


        mout = torch.cat([drug1, drug2], dim=1)


        scores, t = self.mlp_pre(mout)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index, t



    def test(self,output,label):


        positive_class_probs = F.sigmoid(output)

        positive_class_probs = positive_class_probs.detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)

        aupr = average_precision_score(targets, positive_class_probs)
        positive_class_probs = positive_class_probs.flatten()
        targets = targets.flatten()
        positive_class_probs = (positive_class_probs > 0.65).astype(int)
        accuracy = accuracy_score(targets, positive_class_probs)


        precision = precision_score(targets, positive_class_probs)
        recall = recall_score(targets, positive_class_probs)
        return auc,aupr,accuracy,precision,recall

class SliceAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(SliceAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, D, H, W], D=6 is number of slices
        B, C, D, H, W = x.shape

        # Split into 3 parts along D (assume D=6)
        x1 = x[:, :, 0:2, :, :]
        x2 = x[:, :, 2:4, :, :]
        x3 = x[:, :, 4:6, :, :]

        def process_part(part):
            B, C, D, H, W = part.shape
            out = part.view(B, C, D * H * W).transpose(1, 2)  # [B, N, C]
            out, _ = self.attn(out, out, out)
            return out.transpose(1, 2).view(B, C, D, H, W)

        x1 = process_part(x1)
        x2 = process_part(x2)
        x3 = process_part(x3)

        # Concatenate along D (depth)
        return torch.cat([x1, x2, x3], dim=2)  # [B, C, D=6, H, W]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 16, 4, 256, 256)
            nn.ReLU(True),
            nn.Conv3d(16, 32, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 32, 2, 128, 128)
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 64, 1, 64, 64)
            nn.ReLU(True)
        )

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1),  # (32, 128, 1, 32, 32)
            nn.ReLU(True)
        )

        # Attention block after bottleneck
        self.attn_block = SliceAttentionBlock(embed_dim=128, num_heads=4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 4, kernel_size=3, stride=(1, 2, 2), padding=1, output_padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck_conv(x)
        x = self.attn_block(x)  # 融合 attention 后的编码表示
        x_recon = self.decoder(x)
        return x, x_recon

class GateHead(nn.Module):
    def __init__(self, in_dim=256, negative_slope=0.2):
        super(GateHead, self).__init__()
        # Modality A 专属
        self.lin1_a = nn.Linear(in_dim, 128)
        self.lin2_a = nn.Linear(128,    64)
        self.lin3_a = nn.Linear(64,     32)
        # Modality B 专属
        self.lin1_b = nn.Linear(in_dim, 128)
        self.lin2_b = nn.Linear(128,    64)
        self.lin3_b = nn.Linear(64,     32)
        # Modality C 专属
        self.lin1_c = nn.Linear(in_dim, 128)
        self.lin2_c = nn.Linear(128,    64)
        self.lin3_c = nn.Linear(64,     32)
        self.input_norm = nn.LayerNorm(in_dim)
        # 换成 LeakyReLU
        self.act = nn.LeakyReLU(negative_slope)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, negative_slope):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming uniform 适配 LeakyReLU
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=negative_slope,
                    nonlinearity='leaky_relu'
                )
                nn.init.zeros_(m.bias)
    def forward(self, x1, x2, x3):
        eps = 1e-6
        x1 = (x1 - x1.mean(dim=1,keepdim=True)) / (x1.std(dim=1,keepdim=True) + eps)
        x2 = (x2 - x2.mean(dim=1,keepdim=True)) / (x2.std(dim=1,keepdim=True) + eps)
        x3 = (x3 - x3.mean(dim=1,keepdim=True)) / (x3.std(dim=1,keepdim=True) + eps)
        # --- 支路 A ---
        a = self.act(self.lin1_a(x1))
        a = self.act(self.lin2_a(a))
        a = self.act(self.lin3_a(a))
        a = self.pool(a.unsqueeze(1)).squeeze(1)

        # --- 支路 B ---
        b = self.act(self.lin1_b(x2))
        b = self.act(self.lin2_b(b))
        b = self.act(self.lin3_b(b))
        b = self.pool(b.unsqueeze(1)).squeeze(1)

        # --- 支路 C ---
        c = self.act(self.lin1_c(x3))
        c = self.act(self.lin2_c(c))
        c = self.act(self.lin3_c(c))
        c = self.pool(c.unsqueeze(1)).squeeze(1)

        # 拼接后 softmax
        logits = torch.cat([a, b, c], dim=1)  # [B, 3]
        weights = self.softmax(logits)
        return weights

class MolVisGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels2, out_channels_gat,
                 out_channels, global_dim, num_layers, heads, ff_dropout,
                 attn_dropout, spatial_size, skip, dist_count_norm, conv_type, num_centroids, no_bn, norm_type):
        super(MolVisGNN, self).__init__()

        self.conv1 = SAGEConv(256, 128)
        # self.conv11 = SAGEConv(128, 64)
        self.conv2 = SAGEConv(128, 64)
        self.conv3 = SAGEConv(64, 32)
        self.conv4 = GCNConv(16, 8)
        self.con5 = GCNConv(32, 16)
        self.gat = GATConv(256, 64, heads=4, concat=True)
        self.gat2 = GATConv(256, 128, heads=4, concat=False)
        self.jk_linear = nn.Linear(64 + 32 + 16, 32)  # 输出维度可调
        self.Autoencoder = Autoencoder()
        self.Autoencoder2 = ConvAutoencoder()
        self.sp = Directional3DProcessor(128, 64)
        self.con = conv2d()
        self.lne = nn.Linear(768, 128)
        self.mlp_pre = mlp_pre(64, 32, 16, 1)
        self.mlp_pre2 = mlp_pre(32, 16, 8, 1)
        self.nor = torch.nn.BatchNorm1d(128)
        self.nor2 = torch.nn.BatchNorm1d(64)
        self.nor3 = torch.nn.BatchNorm1d(32)
        self.resnet = Conv1dNetwork()
        self.relu = torch.nn.LeakyReLU(0.3)
        self.relu2 = torch.nn.ReLU()
        self.tahn = torch.nn.Tanh()
        self.aaa = aaaa()
        self.gate = GateHead()
        self.fc = torch.nn.Linear(2, 1)

        self.intermediate_feature = None  # 用于存储中间特征
        self.intermediate_gradient = None  # 用于存储梯度

    def forward(self, batch, drug_1d_features, drug_2d_features, drug_3d_features):
        encode_3d = drug_3d_features
        encode_3d = self.sp(encode_3d)
        encode_3d = encode_3d.mean(dim=[2, 3, 4])
        encode_3d = F.adaptive_avg_pool1d(encode_3d, 256)

        # # 融合***************************
        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 256)
        drug_1d_features = F.adaptive_avg_pool1d(drug_1d_features, 256)

        # drug_feature = torch.cat((drug_1d_features,drug_2d_features),dim=1)

        weights = self.aaa(drug_1d_features, drug_2d_features, encode_3d)
        self.saved_x = weights
        w1 = weights[:, 0].unsqueeze(1)
        w2 = weights[:, 1].unsqueeze(1)
        w3 = weights[:, 2].unsqueeze(1)
        drug_2d_features = drug_2d_features * w1
        drug_1d_features = drug_1d_features * w2
        drug_3d_features = encode_3d * w3
        drug_feature = torch.cat((drug_1d_features, drug_2d_features, drug_3d_features), dim=1)
        drug_feature = self.lne(drug_feature)
        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 128)
        # drug_feature = self.nor(drug_feature)
        drug_res = self.resnet(drug_feature)
        drug_feature = drug_res + drug_feature
        # drug_feature = torch.cat((drug_res , drug_feature),dim=1)

        # **************************************************

        # only 3d
        # drug_feature = encode_3d
        # drug_feature = F.adaptive_avg_pool1d(drug_3d_features, 256).squeeze(0)

        # only 1d
        # drug_feature = drug_1d_features
        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 256).squeeze(0)

        # drug_feature = F.adaptive_avg_pool1d(drug_feature, 256).squeeze(0)
        edge_index = batch.edge_index.to(device)
        drug_feature = F.adaptive_avg_pool1d(drug_feature, 256)
        x_dict = self.conv1(drug_feature, edge_index)
        x_dict = self.nor(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv2(x_dict, edge_index)
        x_dict = self.nor2(x_dict)
        x_dict = self.relu(x_dict)

        x_dict = self.conv3(x_dict, edge_index)
        x_dict = self.nor3(x_dict)
        x_dict = self.relu(x_dict)

        # x_dict = self.conv4(x_dict, edge_index)
        # x_dict = self.relu(x_dict)

        # x_dict = self.nor3(x_dict)
        # x_dict = self.gat(drug_feature, edge_index)
        # x_dict = self.relu2(x_dict)
        # x_dict = self.nor2(x_dict)
        # x_dict = self.gat2(x_dict, edge_index)
        # x_dict = self.relu2(x_dict)

        # print(decoder.shape)
        return x_dict, self.saved_x

    def compute_loss(self, out, batch):
        edge_index = batch.edge_label_index
        labels = batch.edge_label

        # 直接使用全部样本（假设正负样本已平衡）
        src = edge_index[0]
        dst = edge_index[1]

        # print(out.shape)
        drug1 = out[src]  # shape [N, dim]
        drug2 = out[dst]  # shape [N, dim]

        # 拼接方式构造边特征
        mout = torch.cat([drug1, drug2], dim=1)  # shape [N, 2*dim]

        # 前向分类
        scores, t = self.mlp_pre(mout)
        # print(scores.shape)
        # 计算损失
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores.squeeze(), labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index, t

    def compute_loss2(self, out, batch):
        # 获取正负边对
        edge_index = batch.edge_label_index  # [2, N]
        labels = batch.edge_label  # [N]

        # 拆分出 drug1, drug2 的节点嵌入
        drug1 = out[edge_index[0]]  # [N, D]
        drug2 = out[edge_index[1]]  # [N, D]

        # —— 用内积来计算 logits scores ——
        # scores[i] = drug1[i] · drug2[i]
        scores = (drug1 * drug2).sum(dim=1)  # [N]

        # 二分类的 BCE + logits
        loss = F.binary_cross_entropy_with_logits(
            scores,
            labels.float().to(scores.device)
        )

        return loss, scores, labels, edge_index

    def test(self, output, label):

        # predictions = F.softmax(output, dim=1)
        positive_class_probs = F.sigmoid(output)
        # positive_class_probs = predictions[:, 1]  # 提取正类的概率
        positive_class_probs = positive_class_probs.detach().cpu().numpy()
        targets = label.cpu().numpy()

        # 计算AUC
        auc = roc_auc_score(targets, positive_class_probs)

        # 计算AUPR
        aupr = average_precision_score(targets, positive_class_probs)
        positive_class_probs = positive_class_probs.flatten()
        targets = targets.flatten()
        # df = pd.DataFrame({'prob': positive_class_probs, 'label': targets})

        # df.to_csv('graph/out.csv', index=False)
        # 计算Accuracy
        positive_class_probs = (positive_class_probs > 0.5).astype(int)
        accuracy = accuracy_score(targets, positive_class_probs)

        # 计算Precision
        precision = precision_score(targets, positive_class_probs)
        recall = recall_score(targets, positive_class_probs)
        return auc, aupr, accuracy, precision, recall

    def casepre(self, out, node_to_idx):
        # out 是每个节点的特征，node_to_idx 是从节点名到索引的映射
        # 获取 DB00619 的索引
        db00619_idx = node_to_idx['DB00619']  # 使用 node_to_idx 获取 DB00619 的索引

        # 获取 DB00619 的特征
        db00619_feature = out[db00619_idx]  # 这是 DB00619 的特征向量

        # 计算 DB00619 与其他所有节点的关系
        scores = []
        node_names = []  # 用于保存与 DB00619 的关系的节点名称
        for idx, node_feature in enumerate(out):
            if idx != db00619_idx:
                # 可以通过拼接 DB00619 的特征与其他节点的特征，或者直接相减/相加
                combined_feature = torch.cat((db00619_feature, node_feature), dim=0)  # 拼接两个特征
                score = self.mlp_pre(combined_feature)  # 使用 MLP 计算得分
                scores.append(score)
                node_names.append(list(node_to_idx.keys())[list(node_to_idx.values()).index(idx)])  # 获取节点名称

        # 将所有得分拼接成一个 tensor
        scores = torch.cat(scores, dim=0)
        scores = torch.sigmoid(scores)
        # 将得分与对应的节点名称一起保存到 CSV 文件
        results_df = pd.DataFrame({
            'node_name': node_names,
            'score': scores.cpu().numpy()  # 转换为 NumPy 数组以便保存
        })

        results_df.to_csv('graph_DDI/db00619_relationship_scores.csv', index=False)

        return scores, results_df  # 返回得分和保存的 DataFrame

