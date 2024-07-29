import torch
from torch import nn
import torch_sparse

from base_classes import ODEFunc
from utils import MaxNFEException

class NODFunc(ODEFunc):
    def __init__(self, in_features, out_features, opt, data, device):
        super(NODFunc, self).__init__(opt, data, device)

        self.in_features = in_features
        self.out_features = out_features
        self.opinion_dim = opt['hidden_dim']



        #self.d = nn.Parameter(torch.Tensor(data.x.shape[0], opt['hidden_dim']))
        self.d = nn.Parameter(torch.Tensor(data.x.shape[0], 1))
        self.alpha = nn.Parameter(torch.Tensor(data.x.shape[0], 1))
        #self.alpha = nn.Parameter(torch.Tensor(data.x.shape[0], opt['hidden_dim']))
        self.u = nn.Parameter(torch.Tensor(data.x.shape[0], 1))


        # Opinion graph parameters
        self.oGraph_upper = nn.Parameter(torch.randn(int(opt['hidden_dim'] * (opt['hidden_dim'] - 1) / 2)))
        self.oGraph_lower = nn.Parameter(torch.randn(int(opt['hidden_dim'] * (opt['hidden_dim'] - 1) / 2)))
        self.oGraph_upper_index = torch.triu_indices(opt['hidden_dim'], opt['hidden_dim'], 1)#.to(device)
        self.oGraph_lower_index = torch.tril_indices(opt['hidden_dim'], opt['hidden_dim'], -1)#.to(device)

    def calc_oGraph(self):
        o_graph = torch.zeros(self.opinion_dim, self.opinion_dim)#.to(self.device)
        o_graph[self.oGraph_upper_index[0], self.oGraph_upper_index[1]] = self.oGraph_upper
        o_graph[self.oGraph_lower_index[0], self.oGraph_lower_index[1]] = self.oGraph_lower
        return o_graph

    """
    def sparse_multiply(self, x):
        if self.opt['block'] in ['attention']:
            mean_attention = self.attention_weights.mean(dim=1)
            ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
        elif self.opt['block'] in ['mixed', 'hard_attention']:
            ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
        else:
            ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
        return ax
    """

    def forward(self, t, x):
        if self.nfe > self.opt["max_nfe"]:
            raise MaxNFEException
        self.nfe += 1

        print(self.x0.shape)
        print(self.in_features)
        print(self.out_features)
        print(self.opinion_dim)

        o_graph = self.calc_oGraph()

        Aax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)

        f = -3*(x) + torch.tanh(self.u* (self.alpha*(x) + Aax + torch.matmul(x, o_graph.t()) + torch.matmul(Aax, o_graph.t())))# + self.x0

        
        return f