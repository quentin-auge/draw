import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.dataset import PADDING_VALUE
from lib.dataset import get_batches


class LSTM(nn.Module):
    def __init__(self, batch_size, n_hidden, n_layers, dropout=0):
        super().__init__()

        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(5, n_hidden, n_layers, dropout=dropout)
        self.output_weights = nn.Linear(n_hidden, 5)

    def forward(self, data, lens):
        self.init_hidden(data.shape[1])

        hidden_state = self.hidden_state
        cell_state = self.cell_state

        packed_data = pack_padded_sequence(data, lens)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_data,
                                                              (hidden_state, cell_state))
        output, _ = pad_packed_sequence(packed_output, padding_value=PADDING_VALUE)

        # Throw away states history
        self.hidden_state = Variable(hidden_state)
        self.cell_state = Variable(cell_state)

        output = self.output_weights(output)

        output = torch.cat([output[:, :, :2],
                            F.softmax(output[:, :, 2:], dim=-1)],
                           dim=2)

        return output

    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros([self.n_layers, batch_size, self.n_hidden])
        self.cell_state = torch.zeros([self.n_layers, batch_size, self.n_hidden])

        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.cuda()
            self.cell_state = self.cell_state.cuda()


def get_reg_loss(preds, labels):
    n_preds = len(preds)
    labels = labels[:n_preds, :, :]

    masked_labels = labels[labels != PADDING_VALUE]
    masked_preds = preds[labels != PADDING_VALUE]

    return F.mse_loss(masked_preds, masked_labels)


def get_classif_loss(preds, labels):
    n_preds = len(preds)
    labels = labels[:n_preds, :, :]

    masked_labels = labels[labels != PADDING_VALUE]
    masked_preds = preds[labels != PADDING_VALUE]

    return F.mse_loss(masked_preds, masked_labels)

    return (1 - masked_preds[masked_labels == 1]).mean()


def get_loss(preds, labels):
    reg_loss = get_reg_loss(preds[:, :, :2], labels[:, :, :2])
    classif_loss = get_classif_loss(preds[:, :, 2:], labels[:, :, 2:])
    return (reg_loss, classif_loss)


def evaluate(model, criterion, ds, mean_stds, batch_size=1024):
    running_reg_loss = 0
    running_classif_loss = 0
    n_batches = 0

    batches = get_batches(ds, mean_stds, batch_size)
    for data_batch, labels_batch, lens_batch in batches:
        preds_batch = model(data_batch, lens_batch)
        reg_loss, classif_loss = criterion(preds_batch, labels_batch)
        running_reg_loss += reg_loss.item()
        running_classif_loss += classif_loss.item()
        n_batches += 1

    reg_loss = running_reg_loss / n_batches
    classif_loss = running_classif_loss / n_batches

    return (reg_loss, classif_loss)


def train(model, scheduler_or_optimizer, criterion, train_ds, val_ds,
          train_means_stds, val_means_stds,
          batch_size, epochs, epochs_between_evals=1):
    if isinstance(scheduler_or_optimizer, torch.optim.Optimizer):
        optimizer = scheduler_or_optimizer
        scheduler = None
    else:
        scheduler = scheduler_or_optimizer
        optimizer = scheduler.optimizer

    for epoch in range(1, epochs + 1):

        train_batches = get_batches(train_ds, train_means_stds, batch_size)
        for data_batch, labels_batch, lens_batch in train_batches:
            preds_batch = model(data_batch, lens_batch)
            optimizer.zero_grad()
            reg_loss, classif_loss = criterion(preds_batch, labels_batch)
            loss = reg_loss + classif_loss
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, criterion, train_ds, train_means_stds, batch_size)
        train_reg_loss, train_classif_loss = train_loss
        train_loss = train_reg_loss + train_classif_loss

        val_loss = evaluate(model, criterion, val_ds, val_means_stds, batch_size)
        val_reg_loss, val_classif_loss = val_loss
        val_loss = val_reg_loss + val_classif_loss

        if scheduler:
            scheduler.step(val_loss)

        if epoch == 1 or epoch % epochs_between_evals == 0 or epoch == epochs:
            loss_ratio = (train_reg_loss + val_reg_loss) / (train_classif_loss + val_classif_loss)

            print(f'epoch: {epoch:3d}'
                  f'   train_loss: {train_loss:.2f}'
                  f'   val_loss: {val_loss:.2f}'
                  f'         reg_classif_loss_ratio: {loss_ratio:.2f}')


def generate(model, start_of_stroke, n_points):
    preds = start_of_stroke

    if torch.cuda.is_available():
        preds = preds.cuda()

    for _ in range(n_points):
        new_preds = model(preds, [len(preds)])
        new_pred = new_preds[-1].unsqueeze(dim=0)
        preds = torch.cat([preds, new_pred])

    preds = preds.squeeze()

    return preds
