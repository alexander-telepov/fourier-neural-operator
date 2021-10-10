import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from timeit import default_timer


def lp_loss_relative(true, pred, p=2, reduction='mean'):
    assert reduction in ['mean', 'sum', 'none']
    assert true.ndim == pred.ndim

    dims = tuple(range(1, true.ndim))
    diff_norm = torch.norm(true - pred, p=p, dim=dims)
    true_norm = torch.norm(true, p=p, dim=dims)

    if reduction == 'mean':
        return torch.mean(diff_norm / true_norm)
    elif reduction == 'sum':
        return torch.sum(diff_norm / true_norm)
    elif reduction == 'none':
        return diff_norm / true_norm


class Trainer:
    def __init__(self, args, net, optimizer, scheduler, train_loader, val_loader, test_loader, criterion=lp_loss_relative):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.unpack_args(args)

        predictive_mode = self.predictive_mode
        if predictive_mode == 'one_step':
            basic_step = self.one_step_prediction
            if self.pad_coordinates == 'true':
                self.prepare_grids()
        elif predictive_mode == 'multiple_step':
            basic_step = self.multiple_step_prediction
        elif predictive_mode == 'unet_step':
            basic_step = self.unet_step
        else:
            raise ValueError(f'Unsupported predictive mode: {predictive_mode}')

        self.basic_step = basic_step
        self.writer = SummaryWriter(os.path.join(self.experiments, self.exp_name, 'tensorboard'))

    def unpack_args(self, args):
        for key, value in args.items():
            setattr(self, key, value)

    def prepare_grids(self):
        S = self.S
        gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
        gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
        gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
        self.gridx = gridx.to(self.device)
        self.gridy = gridy.to(self.device)

    def one_step_prediction(self, xx, yy):
        step, T = self.step, self.T
        loss = 0
        for t in range(0, T, step):
            label = yy[..., t:t+step]
            predict_t = self.net(xx)
            loss += self.criterion(label, predict_t)

            if t == 0:
                predict = label
            else:
                predict = torch.cat((predict, predict_t), -1)

            if self.pad_coordinates == 'true':
                xx = torch.cat((xx[..., step:-2], predict_t,
                                self.gridx.repeat([self.batch_size, 1, 1, 1]),
                                self.gridy.repeat([self.batch_size, 1, 1, 1])),
                               dim=-1)
            else:
                xx = torch.cat((xx[..., step:-2], predict_t), dim=-1)

        return loss, predict

    def multiple_step_prediction(self, inputs, labels):
        predictions = self.net(inputs)
        loss = self.criterion(labels, predictions)
        return loss, predictions

    def unet_step(self, inputs, labels):
        predictions = torch.empty_like(labels)
        if self.training and self.unet_mode == 'robust':
            for t in range(self.t_out):
                predictions[..., t] = self.net(inputs[..., t].unsqueeze(1)).squeeze()
        else:
            predictions[..., 0] = self.net(inputs[..., 0].unsqueeze(1)).squeeze()
            for t in range(1, self.t_out):
                predictions[..., t] = self.net(predictions[..., t-1].unsqueeze(1)).squeeze()

        loss = self.criterion(labels, predictions)

        return loss, predictions

    def train_step(self, inputs, labels):
        self.net.train()
        self.training = True
        loss, predict = self.basic_step(inputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def test(self, dataloader):
        self.net.eval()
        self.training = False
        test_l2_step = 0
        test_l2_full = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, predictions = self.basic_step(inputs, labels)
            test_l2_step += loss
            test_l2_full += self.criterion(labels, predictions)

        return test_l2_full

    def train(self):
        n_train, n_test = len(self.train_loader), len(self.val_loader)
        for epoch in range(self.n_epochs):
            t1 = default_timer()
            loss_train = 0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss_train += self.train_step(inputs, labels)

            loss_val = self.test(self.val_loader)
            t2 = default_timer()
            self.scheduler.step()
            epoch_time = t2-t1
            self.writer.add_scalar('time', epoch_time, epoch)
            self.writer.add_scalar('train_loss', loss_train.item() / n_train, epoch)
            self.writer.add_scalar('val_loss', loss_val.item() / n_test, epoch)
            print(f'Epoch: {epoch} time: {epoch_time}, train_loss: {loss_train.item() / n_train}, val_loss: {loss_val.item() / n_test}')

        self.save_model(epoch)

    def save_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.experiments, self.exp_name, 'model.pth'))

    def load_model(self):
        checkpoint = torch.load(os.path.join(self.experiments, self.exp_name, 'model.pth'), map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return epoch

    @torch.no_grad()
    def predict(self, dataloader):
        dir_ = os.path.join(self.experiments, self.exp_name, 'predictions', self.predictions_path)
        ids, l, bs = dataloader.dataset.ids, dataloader.dataset.l, dataloader.batch_size
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            _, predictions = self.basic_step(inputs, labels)
            for j in range(bs):
                np.save(os.path.join(dir_, f'prediction_{str(ids[bs * i + j]).rjust(l, "0")}.npy'),
                        predictions[j].cpu().numpy())
                np.save(os.path.join(dir_, f'groundtruth_{str(ids[bs * i + j]).rjust(l, "0")}.npy'),
                        labels[j].cpu().numpy())
