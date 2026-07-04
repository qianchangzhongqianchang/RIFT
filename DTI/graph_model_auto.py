import torch.nn as nn
from torch_geometric.nn import GCNConv,HeteroConv,SAGEConv,GraphConv,GATConv
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score,recall_score
import math
from config import device
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

class Conv1dNetwork(nn.Module):
    def __init__(self):
        super(Conv1dNetwork, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)

        self.conv2 = nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1)

        self.conv3 = nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1)

        self.pool = nn.MaxPool1d(4)
        self.pool2 = nn.AdaptiveAvgPool1d(128)
        self.pool3 = nn.AdaptiveMaxPool1d(512)
        self.nor = nn.BatchNorm1d(128)
    def forward(self, x):

        x = x.unsqueeze(1) 

        x = F.relu(self.conv1(x))  
        x = self.pool(x)  

        x = F.relu(self.conv2(x))  
        x = self.pool(x)  

        x = F.relu(self.conv3(x))  
        x = self.pool(x)  
  
        x = x.view(x.size(0), -1) 
        x = x.unsqueeze(1) 
        x = self.pool2(x)
        x = x.squeeze(1)
        x = self.nor(x)
        return x
    

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


class Autoencoder2(nn.Module):
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
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)

    def forward(self, x1,x2,x3):
        x1 = F.relu(self.lin1(x1))
        x2 = F.relu(self.lin1(x2))
        x3 = F.relu(self.lin1(x3))
        x1 = F.relu(self.lin2(x1))
        x2 = F.relu(self.lin2(x2))
        x3 = F.relu(self.lin2(x3))
        x1 = F.relu(self.lin3(x1))
        x2 = F.relu(self.lin3(x2))
        x3 = F.relu(self.lin3(x3))
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
        x = torch.cat((x1,x2,x3),dim=1)
        x = self.softmax(x)
        return x

class TargetConditionedFusion(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()

        self.score_network = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z1, z2, z3, target):
        """
        z1,z2,z3: [E,D]
        target:   [E,D]
        """

        modalities = torch.stack(
            [z1, z2, z3],
            dim=1,
        )  # [E,3,D]

        target_expand = target.unsqueeze(1).expand(
            -1,
            3,
            -1,
        )

        interaction_features = torch.cat(
            [
                modalities,
                target_expand,
                modalities * target_expand,
                torch.abs(modalities - target_expand),
            ],
            dim=-1,
        )  # [E,3,4D]

        logits = self.score_network(
            interaction_features
        ).squeeze(-1)  # [E,3]

        weights = torch.softmax(logits, dim=1)

        fused = (
            modalities * weights.unsqueeze(-1)
        ).sum(dim=1)

        return fused, weights

class mlp_pre(torch.nn.Module):
    def __init__(self, num_in ,num_hid1 , num_hid2 ,num_hid3,num_hid4 ,num_out ):
        super(mlp_pre, self).__init__()
        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.l3 = torch.nn.Linear(num_hid2, num_hid3)
        self.l4 = torch.nn.Linear(num_hid3, num_hid4)
        self.classify = torch.nn.Linear(num_hid4, num_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)
        self.nor = torch.nn.BatchNorm1d(num_hid1)
        self.nor2 = torch.nn.BatchNorm1d(num_hid2)
        self.nor3 = torch.nn.BatchNorm1d(8)
        self.nor4 = torch.nn.BatchNorm1d(4)
        
        # self.nor2 = torch.nn.BatchNorm1d(num_hid2)
    def forward(self, x):
        
        x = self.l1(x)

        x = self.drop(x)
        x = self.l2(x)

        x = self.drop(x)
        x = self.l3(x)


        return x

class EdgePredictor(nn.Module):
    def __init__(self, input_dim=640):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)

class Directional3DProcessor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Directional3DProcessor, self).__init__()

        self.conv_fr = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_bb = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_tl = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoded_3d):  # [B, 128, 6, 32, 32]
        fr = encoded_3d[:, :, 0:2]  # [B, 128, 2, 32, 32]
        bb = encoded_3d[:, :, 2:4]
        tl = encoded_3d[:, :, 4:6]

        fr_out = self.conv_fr(fr)
        bb_out = self.conv_bb(bb)
        tl_out = self.conv_tl(tl)


        combined = torch.cat([fr_out, bb_out, tl_out], dim=2) 
        return combined

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

        # 六个视角共用同一个编码器
        self.view_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        # 六个方向的位置编码
        self.view_embedding = nn.Parameter(
            torch.randn(1, 6, hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # 使用一个可学习查询对六个视角进行聚合
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [B, C, 6, H, W]
        return:
            molecular_feature: [B, hidden_dim]
            view_attention: [B, 6]
        """
        if x.dim() != 5 or x.size(2) != 6:
            raise ValueError(
                f"Expected [B,C,6,H,W], but got {tuple(x.shape)}"
            )

        batch_size, channels, views, height, width = x.shape

        # [B,C,6,H,W] -> [B,6,C,H,W] -> [B*6,C,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size * views, channels, height, width)

        # 每个视角独立编码
        tokens = self.view_encoder(x).flatten(1)
        tokens = tokens.view(batch_size, views, -1)

        # 加入方向信息
        tokens = tokens + self.view_embedding

        # 六个视角相互交流
        tokens = self.transformer(tokens)

        # 自动聚合重要视角
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, attention = self.attention_pool(
            query,
            tokens,
            tokens,
            need_weights=True,
        )

        feature = self.norm(pooled.squeeze(1))
        attention = attention.squeeze(1)

        return feature, attention

ROTATION_PERMUTATIONS = torch.tensor([
    [0, 1, 2, 3, 4, 5],  # 原始方向

    # 绕竖直轴旋转
    [2, 3, 1, 0, 4, 5],
    [1, 0, 3, 2, 4, 5],
    [3, 2, 0, 1, 4, 5],

    # 绕左右轴旋转
    [4, 5, 2, 3, 1, 0],
    [1, 0, 2, 3, 5, 4],
    [5, 4, 2, 3, 0, 1],

    # 绕前后轴旋转
    [0, 1, 4, 5, 3, 2],
    [0, 1, 3, 2, 5, 4],
    [0, 1, 5, 4, 2, 3],
], dtype=torch.long)


    
class RIFT(nn.Module):
    def __init__(self):
        super().__init__()

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

        self.drug_1d_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.drug_2d_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.drug_3d_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.graph_drug_projector = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )

        self.conv1 = HeteroConv({
            ("Protein", "interacts", "drug"):
                SAGEConv(256, 128),
            ("drug", "interacts", "Protein"):
                SAGEConv(256, 128),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("Protein", "interacts", "drug"):
                SAGEConv(128, 64),
            ("drug", "interacts", "Protein"):
                SAGEConv(128, 64),
        }, aggr="mean")

        self.conv3 = HeteroConv({
            ("Protein", "interacts", "drug"):
                SAGEConv(64, 32),
            ("drug", "interacts", "Protein"):
                SAGEConv(64, 32),
        }, aggr="mean")

        self.activation = nn.LeakyReLU(0.2)

        self.drug_context_projector = nn.Linear(32, 128)
        self.target_context_projector = nn.Linear(32, 128)

        self.target_fusion = TargetConditionedFusion(
            hidden_dim=128,
            dropout=0.1,
        )

        self.edge_predictor = EdgePredictor(
            input_dim=640,
        )
    def forward(self, x_dict, edge_index_dict,drug_2d_features, drug_3d_features):
        # encode_3d = drug_3d_features
        # encode_3d = self.sp(encode_3d)
        # encode_3d = encode_3d.mean(dim=[2, 3, 4])
        # encode_3d = F.adaptive_avg_pool1d(encode_3d, 128)

        z_3d_raw, view_attention = self.view_encoder(drug_3d_features)
        z_3d = self.drug_3d_projector(z_3d_raw)
        if self.training:
            rotated_views = self.rotate_views(drug_3d_features)

            z_3d_rotated_raw, _ = self.view_encoder(
                rotated_views
            )
            z_3d_rotated = self.drug_3d_projector(
                z_3d_rotated_raw
            )
            rotation_loss = (
                    1.0
                    - F.cosine_similarity(
                z_3d,
                z_3d_rotated,
                dim=-1,
            ).mean()
            )
        else:
            rotation_loss = z_3d.new_tensor(0.0)

        drug_2d_features = F.adaptive_avg_pool1d(drug_2d_features, 128).squeeze(0)
        x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 128).squeeze(0)
        drug_1d_features = x_dict['drug']
        z_1d = self.drug_1d_projector(drug_1d_features)
        z_2d = self.drug_2d_projector(drug_2d_features)
        #z_3d = self.drug_3d_projector(z_3d)
        drug_graph_input = self.graph_drug_projector(
            torch.cat([z_1d, z_2d, z_3d], dim=-1)
        )
        # x = self.gate(x_dict['drug'],drug_2d_features, encode_3d)
        # self.saved_x = x.detach().cpu().numpy()
        # drug_2d_features = drug_2d_features * x[:,1].unsqueeze(1)
        # drug_1d_features = drug_1d_features * x[:,0].unsqueeze(1)
        # encode_3d = encode_3d * x[:,2].unsqueeze(1)
        #
        # x_dict['drug'] = torch.cat((drug_1d_features,drug_2d_features,encode_3d), dim=1)
        #
        # x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'].unsqueeze(0), 128).squeeze(0)
        # x_dict['drug'] = self.nor3(x_dict['drug'])
        #
        # drug_res = self.resnet(x_dict['drug'])
        #
        #
        # x_dict['drug'] = drug_res + x_dict['drug']



        # x_dict['drug'] = F.adaptive_avg_pool1d(x_dict['drug'], 256).squeeze(0)
        # x_dict['Protein'] = F.adaptive_avg_pool1d(x_dict['Protein'].unsqueeze(0), 256).squeeze(0)
        drug_1d_features = F.adaptive_avg_pool1d(
            x_dict["drug"],
            128,
        )

        protein_graph_input = F.adaptive_avg_pool1d(
            x_dict["Protein"],
            256,
        )
        protein_graph_input=F.adaptive_avg_pool1d(x_dict['Protein'].unsqueeze(0), 256).squeeze(0)
        graph_x_dict = {
            "drug": drug_graph_input,
            "Protein": protein_graph_input,
        }

        graph_x = {
            "drug": drug_graph_input,
            "Protein": protein_graph_input,
        }

        graph_x = self.conv1(graph_x, edge_index_dict)
        graph_x = {
            key: self.activation(value)
            for key, value in graph_x.items()
        }

        graph_x = self.conv2(graph_x, edge_index_dict)
        graph_x = {
            key: self.activation(value)
            for key, value in graph_x.items()
        }

        graph_x = self.conv3(graph_x, edge_index_dict)
        graph_x = {
            key: self.activation(value)
            for key, value in graph_x.items()
        }
        # x_dict = self.conv1(x_dict, edge_index_dict)
        # x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        # x_dict = self.conv2(x_dict, edge_index_dict)
        # x_dict = {key:self.relu(x) for key, x in x_dict.items()}
        #
        # x_dict = self.conv3(x_dict, edge_index_dict)
        # x_dict = {key:self.relu(x) for key, x in x_dict.items()}

        return {
            "z_1d": z_1d,
            "z_2d": z_2d,
            "z_3d": z_3d,
            "graph_drug": graph_x["drug"],
            "graph_protein": graph_x["Protein"],
            "rotation_loss": rotation_loss,
            "view_attention": view_attention,
        }


    # def compute_loss(self,out, batch):
    #
    #     # 获取边
    #     edge_index = batch[('drug', 'interacts', 'Protein')].edge_label_index
    #     # 标签
    #     labels =  batch[('drug', 'interacts', 'Protein')].edge_label
    #     scoreout = []
    #     for d,m in zip(edge_index[0],edge_index[1]) :
    #
    #         Protein_feature = out['Protein'][m]
    #         drug_feature = out['drug'][d]
    #         edge_feature = torch.cat((drug_feature, Protein_feature ), dim=0)
    #
    #
    #         scoreout.append(edge_feature)
    #     scoreout = torch.stack(scoreout)
    #
    #     scoreout = self.mlp_pre(scoreout)
    #
    #     edge = edge_index
    #
    #     scores = scoreout.to(device)
    #     labels = labels.to(device)
    #     scores = scores.squeeze(1)
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels.float())
    #
    #
    #     total_loss =  loss
    #
    #     return total_loss,scores, labels,edge
    def rotate_views(self, x):
        """
        x: [B,C,6,H,W]
        每个药物随机选择一个旋转排列
        """
        batch_size, channels, views, height, width = x.shape

        random_ids = torch.randint(
            0,
            self.rotation_permutations.size(0),
            (batch_size,),
            device=x.device,
        )

        permutations = self.rotation_permutations[random_ids]

        x_view = x.permute(0, 2, 1, 3, 4)

        gather_index = permutations[:, :, None, None, None].expand(
            -1, -1, channels, height, width
        )

        x_rotated = torch.gather(
            x_view,
            dim=1,
            index=gather_index,
        )

        return x_rotated.permute(0, 2, 1, 3, 4).contiguous()

    def decode_edges(self, outputs, edge_index):
        drug_index = edge_index[0]
        protein_index = edge_index[1]

        z1_edge = outputs["z_1d"][drug_index]
        z2_edge = outputs["z_2d"][drug_index]
        z3_edge = outputs["z_3d"][drug_index]

        drug_context = self.drug_context_projector(
            outputs["graph_drug"][drug_index]
        )

        protein_context = self.target_context_projector(
            outputs["graph_protein"][protein_index]
        )

        interaction_drug, modality_weights = self.target_fusion(z1_edge,z2_edge,z3_edge,protein_context,)

        edge_feature = torch.cat(
            [
                interaction_drug,
                drug_context,
                protein_context,
                interaction_drug * protein_context,
                torch.abs(interaction_drug - protein_context),
            ],
            dim=-1,
        )

        scores = self.edge_predictor(edge_feature).squeeze(-1)

        return scores, modality_weights

    def compute_loss(
            self,
            outputs,
            batch,
            lambda_rotation=0.05,
    ):
        edge_store = batch[
            ("drug", "interacts", "Protein")
        ]

        edge_index = edge_store.edge_label_index
        labels = edge_store.edge_label.float().to(
            edge_index.device
        )

        scores, modality_weights = self.decode_edges(
            outputs,
            edge_index,
        )

        prediction_loss = F.binary_cross_entropy_with_logits(
            scores,
            labels,
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
            "edge_index": edge_index,
            "modality_weights": modality_weights,
        }

    def test(self, output, label):
        positive_class_probs = torch.sigmoid(output).detach().cpu().numpy()
        targets = label.cpu().numpy()


        auc = roc_auc_score(targets, positive_class_probs)
        aupr = average_precision_score(targets, positive_class_probs)

        # 将概率转换为二进制预测
        predicted = (positive_class_probs > 0.5).astype(int)

        # 计算其他指标
        accuracy = accuracy_score(targets, predicted)
        precision = precision_score(targets, predicted, zero_division=0)
        recall = recall_score(targets, predicted, zero_division=0)
        f1 = f1_score(targets, predicted, zero_division=0)
        return auc, aupr, accuracy, precision, recall, f1

