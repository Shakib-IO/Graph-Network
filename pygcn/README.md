### [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

Get the [environment.yml](https://github.com/Shakib-IO/Graph-Network/blob/Basic_GNN/environment.yml)
```
conda update conda
conda env create -f environment.yml
conda activate pyg 
```

```
python train.py
```

```
('---no-cuda', action='store_true', default=False, help = "Disables CUDA training.")
('--fastmode', action='store_true', default=False, help='Validate during training pass.')
('--seed', type=int, default=42, help='Random seed.')
('--epochs', type=int, default=100, help="Number of epochs to train")
('--lr', type=float, default=0.01, help="Initial learning Rate")
('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
('--hidden', type=int, default=16, help='Number of hidden units.')
('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
```
