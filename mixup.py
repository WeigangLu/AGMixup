import os.path as osp

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
import os

class UMixup:
    def __init__(self, alpha=0.1, hop=2, shrink_ratio=1., batch_size=-1, mix_method="agmixup", lam_type="auto", pair_type="random", beta=0.5, gamma=0.5, lam=0., name = "cora", device=None):
        self.alpha = alpha
        self.hop = hop
        self.lam = lam
        self.batch_size = batch_size
        
        # setting for mixup
        self.mix_method = mix_method
        self.lam_type = lam_type
        self.pair_type = pair_type
        self.shrink_ratio = shrink_ratio

        self.name = name
        self.beta = beta
        self.gamma = gamma

        self.indices_cache = None
        self.sub_graphs_cache = []
        self.mixed_graphs_cache = None
        self.device = device

        self._init_lam_()

    def _init_lam_(self):
        if self.lam_type == "beta":
            self.lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
            
    def _init_mixed_graphs_cache_(self, num):
        if self.mixed_graphs_cache is None:
            self.mixed_graphs_cache = []
            for i in range(num):
                self.mixed_graphs_cache.append([])
                for j in range(num):
                    self.mixed_graphs_cache[i].append(None)
    
            
    def adaptive_lam(self, p_i, p_j, x_i, x_j):
        # Calculate uncertainty for nodes i and j
        u_i = -torch.sum(p_i * torch.log2(p_i + 1e-16))
        u_j = -torch.sum(p_j * torch.log2(p_j + 1e-16))

        # Calculate contextual similarity
        s_ij = torch.exp(-self.gamma * torch.sum((x_i - x_j) ** 2))
        # s_ij = torch.exp(-self.gamma * F.cosine_similarity(x_i, x_j))
        
        # lambda initialization
        lambda_base = 0.5 * s_ij
        # uncertainty adjustment
        C = p_i.shape[0]
        lambda_adjust = (u_i - u_j) / np.log(C)

        # final lambda
        lambda_final = torch.clamp(lambda_base + self.beta * lambda_adjust, 0, 1)

        return lambda_final

    def get_k_hop_subgraph(self, node_index, edge_index):
        data_path = "./"
        folder_name = osp.join(data_path, f"./data/{self.name}/subgraphs/hop_{self.hop}")
        if not osp.exists(folder_name):
            os.makedirs(folder_name)
        file_name = osp.join(folder_name, f"{node_index}.pt")
        if not osp.exists(file_name):
            subgraph = k_hop_subgraph(node_index, self.hop, edge_index, relabel_nodes=True)
            torch.save(subgraph, file_name)
        else:
            subgraph = torch.load(file_name)
        return subgraph

    def get_pair_indices(self, num):
        if self.pair_type == "random":
            return torch.randperm(num).to(self.device)
        elif self.pair_type == "static":
            # fix the pair index
            if self.indices_cache is None:
                self.indices_cache = torch.randperm(num).to(self.device)
            return self.indices_cache

    
    def agmixup(self, g, embed=None, pred=None):
        labeled_x, labeled_y = g.x[g.train_mask], g.y[g.train_mask]
        x = g.x
        train_idx = g.train_mask.nonzero().squeeze().to(self.device)
        train_size = g.train_mask.sum()

        if self.shrink_ratio < 1:
            train_size = int(train_size * self.shrink_ratio)
            random_index = torch.randperm(train_size).to(self.device)
            train_idx = train_idx[random_index]
            labeled_y = labeled_y[random_index]

        # print(f"Size for Mixup: {train_size}")

        self._init_mixed_graphs_cache_(train_size)
        indices = self.get_pair_indices(train_size)

        # Mixed subgraphs
        # print("Generating Subgraphs...")
        if len(self.sub_graphs_cache) == train_size:
            mixed_graphs = self.sub_graphs_cache
        else:
            mixed_graphs = [self.get_k_hop_subgraph(node, g.edge_index) for node in train_idx.tolist()]
            self.sub_graphs_cache = mixed_graphs

        train_indices = torch.arange(train_size).tolist()
        batch = [train_indices]
        if self.batch_size > 0:
            batch = []
            batch_num = int(train_size / self.batch_size) if int(train_size / self.batch_size) * self.batch_size >= train_size else int(train_size / self.batch_size) + 1
            for i in range(batch_num - 1):
                batch.append(train_indices[i*self.batch_size:(i+1)*self.batch_size])
            batch.append(train_indices[self.batch_size*(batch_num-1):])
        
        mega_data_list = []
        for batch_idx in batch:
            mixed_data = [[None, 0]] # [subgraph, num_nodes]
            mapping_list = []
            num_nodes = 0
            # print("Start Subgraphs Mixup...")
            for i in batch_idx:
                # l_from_train_id, r_from_train_id = l_from_train_indices[i], r_from_train_indices[i]
                # the edge_index of subgraph has been rearranged from 0
                l_node_index, l_edge_index, l_mapping, _ = mixed_graphs[i]
                r_node_index, r_edge_index, r_mapping, _ = mixed_graphs[indices[i]] 
                l_edge_index, r_edge_index = l_edge_index.clone(), r_edge_index.clone()

                l_y = labeled_y[i].unsqueeze(0)
                r_y = labeled_y[indices[i]].unsqueeze(0)
                l_node_index, l_edge_index, l_mapping, r_node_index, r_edge_index, r_mapping, l_y, r_y = l_node_index.to(self.device), l_edge_index.to(self.device), l_mapping.to(self.device), r_node_index.to(self.device), r_edge_index.to(self.device), r_mapping.to(self.device), l_y.to(self.device), r_y.to(self.device)
                
                # for mixing edge index through the central node
                mixed_edge_index = self.mixed_graphs_cache[i][indices[i]]
                if mixed_edge_index is  None:
                    r_central_node_mask = r_edge_index == r_mapping #r_node_index[r_mapping]
                    if l_edge_index.numel() > 0:
                        r_edge_index += l_edge_index.max()
                    r_edge_index[r_central_node_mask] = l_mapping #l_node_index[l_mapping]
                    mixed_edge_index = torch.cat([l_edge_index, r_edge_index], dim=1)
                    self.mixed_graphs_cache[i][indices[i]] = mixed_edge_index

                # generate lambda
                lam = self.lam
                if self.lam_type == "auto":
                    lam = self.adaptive_lam(pred[l_node_index].mean(0), pred[r_node_index].mean(0), embed[l_node_index].mean(0), embed[r_node_index].mean(0))

                # for mixing node features through the central node
                l_x, r_x = x[l_node_index], x[r_node_index]
                l_x[l_mapping] = lam * l_x[l_mapping] + (1 - lam) * r_x[r_mapping]
                mixed_x = torch.cat([l_x, torch.cat([r_x[:r_mapping], r_x[r_mapping + 1:]])])

                sub_g = Data(x=mixed_x, edge_index=mixed_edge_index + num_nodes, l_y=l_y, r_y=r_y, lam=lam)
                mixed_data.append([sub_g, mixed_x.shape[0]])
                mapping_list.append(l_mapping)
                num_nodes += mixed_x.shape[0]

            filter_mixed_data = mixed_data[1:]
            mega_x = torch.cat([data[0].x for data in filter_mixed_data], dim=0).to(self.device)
            mega_l_y = torch.cat([data[0].l_y for data in filter_mixed_data], dim=0).to(self.device)
            mega_r_y = torch.cat([data[0].r_y for data in filter_mixed_data], dim=0).to(self.device)
            mega_lam = torch.Tensor([data[0].lam for data in filter_mixed_data]).to(self.device)
            mega_mapping = torch.cat([mapping + sum([filter_mixed_data[j - 1][0].x.shape[0] for j in range(inner_i, 0, -1)])
                                    for inner_i, mapping in enumerate(mapping_list)], dim=0).to(self.device)

            mega_edge_index = torch.cat([mixed_graph[0].edge_index for mixed_graph in mixed_data[1:]], dim=1).to(self.device)

            mega_data = Data(x=mega_x, edge_index=mega_edge_index, l_y=mega_l_y, r_y=mega_r_y, mixed_index=mega_mapping,
                            lam=mega_lam)
            mega_data_list.append(mega_data)

        return mega_data_list

    def mixup(self, g, embed=None, pred=None):

        mixed_data =  self.agmixup(g, embed, pred)
       
        return mixed_data
