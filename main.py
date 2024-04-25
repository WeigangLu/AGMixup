import argparse
from utils.dataset import Graph
from utils.train import run_training
import torch
import torch.nn as nn
from models.gnn import GNN
import random
import numpy as np
from logger import Logger
import json



def seed_setting(args, i):
    seeds_dict = {
        "cora" : (2870,4439,3067,2743,2830,5608,1288,1743,2698,3431),
        "citeseer" : (2798,1348,3581,2448,2912,1070,2469,1308,1039,1841),
        "pubmed" : (3253,2173,168,1011,1059,503,3006,3098,3194,1897)
    }

    seed = seeds_dict[args.dataset][i]
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        pass

def convert_string(value):
    # -> INT
    try:
        return int(value)
    except ValueError:
        pass
    
    # -> FLOAT
    try:
        return float(value)
    except ValueError:
        pass

    return value

def get_args():
    parser = argparse.ArgumentParser(description='GNN Training and Evaluation')
    parser.add_argument('--model', type=str, default='GCN', help='Name of the GNN model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--hid_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout on features")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay rate')

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--dataset', type=str, default='cora', help='Name of the dataset')
    parser.add_argument('--setting', type=str, default='semi', help='semi or full')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--num_per_class', type=int, default=-1)

    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha for mixup')
    parser.add_argument('--beta', type=float, default=2, help='Scale factor for contextual similarity')
    parser.add_argument('--gamma', type=float, default=3, help='Sensitivity for uncertainty')
    parser.add_argument('--lam', type=float, default=0., help='Static lambda value (active when lam_type = static)')
    parser.add_argument('--mu', type=float, default=1., help='Balance between classification and mixup')
    parser.add_argument('--hop', type=int, default=2, help='Subgraph hop')
    parser.add_argument('--mix_method', type=str, default="agmixup", help='Mixup Method [agmixup, fgmixup, rgmixup, none]')
    parser.add_argument('--lam_type', type=str, default="auto", help='Mixing Parameter Generation [beta, static, auto, none]')
    parser.add_argument('--pair_type', type=str, default="random", help='Mixup Pair Type [random, static]')
    parser.add_argument('--shrink_ratio', type=float, default=1, help='Scale the Size of Mixup Pairs')
    
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--use_param', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--exp_tag', type=str, default="")

    return parser.parse_args()

def trail(args):
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    
    results = []
    logger = Logger(args)
    # for AGMixup's variant
    if args.mix_method == "fgmixup":
        args.lam_type = "static"
    elif args.mix_method == "rgmixup":
        args.lam_type = "beta"

    print(f"All {args.runs} Runs")
    for i in range(args.runs):
        
        seed_setting(args, i)
        
        g = Graph(root='data/', 
                  name=args.dataset, 
                  device=device,
                  num_per_class=args.num_per_class)

        if args.model != "GAT":
            args.heads = 1

        model = GNN(
            in_channels=g.num_features,
            hid_channels=args.hid_channels,
            out_channels=g.num_classes,
            num_layers=args.num_layers,
            dropout=args.dropout,
            layer_name=args.model,
            bias=True,
            bn=args.use_bn,
            heads=args.heads
        ).to(device)
 
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
        criterion = nn.CrossEntropyLoss()

        test_acc = run_training(model, optimizer, criterion, g, args, logger, device)
       
  
        print(f"RUN {i + 1} Final Acc:{test_acc}")
        results.append(test_acc)
        logger.add_result(i, test_acc, "acc")

    print(f"Final Result :${np.mean(results):.2f}_",
          "{", f"\pm{np.std(results):.2f}", "}$ at ",
          f"{args.dataset} with {args.runs} runs",
          sep="")

    logger.print_statistics()

def main():
    args = get_args()
    
    if args.use_param:
        with open('configure.json', 'r') as f:
            config = json.load(f)
        params = config[args.dataset][args.model]
        for key, value in params.items():
            setattr(args, key, convert_string(value))
    
   
    trail(args)

if __name__ == "__main__":
    main()
