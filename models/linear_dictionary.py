import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.code_dim = args.code_dim
        self.im_size = min(args.im_size, args.patch_size) if args.patch_size > 0 else args.im_size
        assert self.im_size > 0
        self.output_dim = (self.im_size ** 2) * self.n_channels
        self.decoder = nn.Linear(self.code_dim, self.output_dim, bias=False)
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

    def forward(self, code):
        output = self.decoder(code)
        return output.view(output.shape[0], self.n_channels, self.im_size, -1)

    def initZs(self, batch_size):
        Zs = torch.zeros(size=(batch_size, self.code_dim), device=self.device)
        return Zs

    def viz_columns(self, n_samples=24, norm_each=False):
        # Visualize columns of linear decoder
        cols = []
        W = self.decoder.weight.data
        max_abs = W.abs().max()
        # Iterate over columns
        for c in range(n_samples):
            column = W[:, c].detach().clone()
            if norm_each:
                max_abs = column.abs().max()
            # Map values to (-0.5, 0.5) interval
            if max_abs > 0:
                column /= (2 * max_abs)
            # Map 0 to gray (0.5)
            column += 0.5
            # Reshape column to output shape
            column = column.view(self.n_channels, self.im_size, -1)
            cols.append(column)
        cols = torch.stack(cols)
        return cols

    def load_pretrained(self, path, freeze=True):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
