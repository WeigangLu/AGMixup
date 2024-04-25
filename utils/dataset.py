from torch_geometric.datasets import Planetoid, Coauthor, CitationFull
import torch
import torch_geometric.transforms as T

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

class Graph(object):
    def __init__(self, name, root, device, num_per_class=-1):
        self.name = name
        self.device = device
        self.dataset = get_dataset(name, root)
        self.data = self.dataset[0]

        self.train_mask = None
        self.test_mask = None
        self.val_mask = None
        
        self.generate_mask(num_per_class)

    @property
    def num_features(self):
        return self.dataset.num_features

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_nodes(self):
        return self.data.num_nodes

    @property
    def x(self):
        return self.data.x.to(self.device)

    @property
    def y(self):
        if self.name == "ogbn-arxiv":
            return self.data.y.squeeze(1).to(self.device)
        return self.data.y.to(self.device)

    @property
    def edge_index(self):
        if self.name == "ogbn-arxiv":
            row, col, _ = self.data.adj_t.to_symmetric().coo()  # data is torch_geometric.data.data.Data
            edge_index = torch.stack([row, col], axis=0)
            return edge_index.to(self.device)
        return self.data.edge_index.to(self.device)


    def generate_mask(self, num_per_class=-1):
        percls_trn = num_per_class if num_per_class > 0 else 20
        
        if self.name in ('cora', 'citeseer', 'pubmed'):
            self.train_mask, self.val_mask, self.test_mask = self.random_splits_planetoid(percls_trn=percls_trn)
        elif self.name in ("coauthor", "physics"):
            self.train_mask, self.val_mask, self.test_mask = self.random_splits_CS(percls_trn=percls_trn)
        elif self.name == "ogbn-arxiv":
            split_idx = self.dataset.get_idx_split()
            train_idx, val_index, test_index = split_idx['train'], split_idx['valid'], split_idx['test']
            self.train_mask = index_to_mask(train_idx, self.num_nodes)
            self.val_mask = index_to_mask(val_index, self.num_nodes)
            self.test_mask = index_to_mask(test_index, self.num_nodes)
        else:
            self.train_mask = self.data.train_mask
            self.test_mask = self.data.test_mask
            self.val_mask = self.data.val_mask
        # else:
        #     self.train_mask, self.test_mask, self.val_mask = self.random_splits(ratio)

    def random_splits_CS(self, percls_trn=20, percls_val=30):
        labels = torch.clone(self.y)
        num_nodes = labels.shape[0]
        indices = []
        for i in range(self.num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

        rest_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=num_nodes)
        val_mask = index_to_mask(val_index, size=num_nodes)
        test_mask = index_to_mask(rest_index, size=num_nodes)

        return train_mask, val_mask, test_mask
    
    def random_splits_planetoid(self, percls_trn=20, val_size=500):
        labels = torch.clone(self.y)
        num_nodes = labels.shape[0]
        indices = []
        for i in range(self.num_classes):
            index = (labels == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        # val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=num_nodes)
        val_mask = index_to_mask(rest_index[:val_size], size=num_nodes)
        test_mask = index_to_mask(rest_index[val_size:], size=num_nodes)

        return train_mask, val_mask, test_mask
    



def get_dataset(name, root='./data'):
    transform = T.NormalizeFeatures()
    if name in ('cora', 'citeseer', 'pubmed'):
        return Planetoid(root, name, transform=transform)
    elif name in ('corafull', 'citeseerfull', 'pubmedfull'):
        name_dict = {}
        return CitationFull(root, "cora", transform=transform)
    elif name in ["coauthor", "physics"]:
        name = "CS" if name != "physics" else "physics"
        return Coauthor(root, name, transform=T.NormalizeFeatures())
    elif name in ['ogbn-arxiv']:
        from ogb.nodeproppred import PygNodePropPredDataset
        return PygNodePropPredDataset(name=name, transform=T.ToSparseTensor())
    else:
        raise ValueError("Dataset not found or not supported.")
