import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Logistics
        self.im_size = min(args.im_size, args.patch_size) if args.patch_size > 0 else args.im_size
        assert self.im_size > 0
        self.n_channels = args.n_channels
        self.code_dim = args.code_dim
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # Architecture
        self.T = args.num_iter_LISTA
        self.W = nn.Linear(self.n_channels * self.im_size ** 2, self.code_dim, bias=True)
        self.W.bias.data.fill_(0)
        self.S = nn.Linear(self.code_dim, self.code_dim, bias=False) # no bias in second layer
        self.relu = nn.ReLU()

    def forward(self, y):
        B = self.W(y.view(y.shape[0], -1))
        Z = self.relu(B)
        # LISTA loop
        for t in range(self.T):
            C = B + self.S(Z)
            Z = self.relu(C)
        return Z.view(Z.shape[0], -1)

    def load_pretrained(self, path, freeze=False):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
