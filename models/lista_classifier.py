from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model
        self.code_dim = args.code_dim
        self.output_dim = args.n_classes
        self.log_interval = args.log_interval
        self.cuda = args.cuda
        self.device = torch.device("cuda" if self.cuda else "cpu")

        # LISTA encoder
        args.n_channels = 1
        self.lista = getattr(import_module('models.{}'.format('lista_encoder')), 'Encoder')(args).to(self.device)

        # Linear classifier on top of codes
        self.classifier = nn.Linear(self.code_dim, self.output_dim, bias=True)

        # Optimization
        self.lr = args.lr
        self.L1_reg = args.L1_reg
        self.L2_reg = args.L2_reg
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        param_groups = [{'params': self.lista.parameters()},
                        {'params': self.classifier.parameters()}]
        self.optimizer = optim.Adam(params=param_groups, lr=self.lr, weight_decay=self.L2_reg)

    def forward(self, input):
        code = self.lista(input)
        output = self.classifier(code)
        return output

    def train(self, data_loader, epoch, k=3):
        self.lista.train()
        self.classifier.train()
        loss_aggr = 0
        top1_acc_aggr = 0
        topk_acc_aggr = 0
        n_samples = 0
        for batch_id, (input, target) in enumerate(data_loader):
            # Data
            input, target = input.to(self.device), target.to(self.device)
            batch_size = input.shape[0]
            input = input.view(batch_size, -1)
            n_samples += batch_size
            total_loss = 0

            # Forward
            self.optimizer.zero_grad()
            self.lista.zero_grad()
            self.classifier.zero_grad()
            output = self.classifier(self.lista(input))

            # Loss and update
            loss_CE = self.criterion(output, target)
            total_loss += loss_CE
            if self.L1_reg > 0:
                l1_norm = self.L1_norm()
                total_loss += self.L1_reg * l1_norm
            total_loss.backward()
            self.optimizer.step()
            loss_aggr += loss_CE.detach() * batch_size

            # Predictions
            top1_pred = output.detach().max(1, keepdim=True)[1]  # get the index of the max log-probability
            top1_pred = top1_pred.eq(target.view_as(top1_pred)).sum().float()
            top1_acc_aggr += top1_pred
            topk_pred = output.detach().topk(k)[1]
            topk_pred = topk_pred.eq(target.unsqueeze(1).expand(-1, k)).sum().float()
            topk_acc_aggr += topk_pred
            if self.log_interval > 0 and batch_id % self.log_interval == 0:
                print('Train Epoch: {} [{}/{}] Loss: {:.6f} Top1: {:.2f} Top{}: {:.2f}'.format(
                    epoch + 1, batch_id + 1, len(data_loader),
                    loss_CE.detach(), top1_pred / batch_size, k, topk_pred / batch_size))

        return {'loss': loss_aggr / n_samples,
                'top1': top1_acc_aggr / n_samples,
                f'top{k}': topk_acc_aggr / n_samples,
                'n_samples': n_samples}


    def test(self, data_loader, k=3):
        self.lista.eval()
        self.classifier.eval()
        with torch.no_grad():
            loss_aggr = 0
            top1_acc_aggr = 0
            topk_acc_aggr = 0
            n_samples = 0
            for batch_id, (input, target) in enumerate(data_loader):
                # Data
                input, target = input.to(self.device), target.to(self.device)
                batch_size = input.shape[0]
                input = input.view(batch_size, -1)
                n_samples += batch_size

                # Forward
                output = self.classifier(self.lista(input))

                # Loss
                loss = self.criterion(output, target)
                loss_aggr += loss.detach() * batch_size

                # Predictions
                top1_pred = output.detach().max(1, keepdim=True)[1]  # get the index of the max log-probability
                top1_pred = top1_pred.eq(target.view_as(top1_pred)).sum().float()
                top1_acc_aggr += top1_pred
                topk_pred = output.detach().topk(k)[1]
                topk_pred = topk_pred.eq(target.unsqueeze(1).expand(-1, k)).sum().float()
                topk_acc_aggr += topk_pred

            # Logging
            loss_aggr = loss_aggr / n_samples
            top1_acc_aggr = top1_acc_aggr / n_samples
            topk_acc_aggr = topk_acc_aggr / n_samples

            return {'loss': loss_aggr,
                    'top1': top1_acc_aggr,
                    f'top{k}': topk_acc_aggr,
                    'n_samples': n_samples}

    def L1_norm(self):
        # Compute the l1 norm of the classifier's weights
        l1_norm = torch.norm(self.classifier.weight, p=1) + torch.norm(self.classifier.bias, p=1)
        l1_norm += torch.norm(self.lista.W.weight, p=1) + torch.norm(self.lista.W.bias, p=1) + torch.norm(self.lista.S.weight, p=1)
        return l1_norm

    def load_pretrained(self, path, freeze=False):
        # Load pretrained model
        pretrained_model = torch.load(f=path, map_location="cuda" if self.cuda else "cpu")
        msg = self.load_state_dict(pretrained_model)
        print(msg)

        # Freeze pretrained parameters
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
