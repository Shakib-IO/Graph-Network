import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gcn1 = GraphConvolution(nfeat, nhid)
        self.gcn2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x,adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim = 1)