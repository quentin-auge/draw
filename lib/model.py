import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.dataset import END_OF_STROKE_VALUE, PADDING_VALUE
from lib.dataset import get_batches

class LSTM(nn.Module):
    def __init__(self, batch_size, n_hidden, n_layers, dropout=0):
        super().__init__()

        self.batch_size = batch_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(2, n_hidden, n_layers, dropout=dropout)
        self.output_weights = nn.Linear(n_hidden, 2)

        self.init_hidden(batch_size)

    def forward(self, data, lens):
        if data.shape[1] != self.batch_size:
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

        output = torch.sigmoid(output) * END_OF_STROKE_VALUE

        return output

    def init_hidden(self, batch_size):
        self.hidden_state = torch.zeros([self.n_layers, batch_size, self.n_hidden])
        self.cell_state = torch.zeros([self.n_layers, batch_size, self.n_hidden])

        if torch.cuda.is_available():
            self.hidden_state = self.hidden_state.cuda()
            self.cell_state = self.cell_state.cuda()


def masked_mse_loss(preds, labels, data):
    preds_len = len(preds)

    flat_preds = preds[:preds_len].view(-1)
    flat_labels = labels[:preds_len].contiguous().view(-1)

    flat_data = data[:preds_len].contiguous().view(-1)
    mask = flat_data != PADDING_VALUE

    return F.mse_loss(flat_preds[mask], flat_labels[mask])


def evaluate(model, ds, criterion, batch_size=1024):
    model.init_hidden(batch_size)

    running_loss = 0
    n_batches = 0

    batches = get_batches(ds, batch_size)
    for data_batch, labels_batch, lens_batch in batches:
        preds_batch = model(data_batch, lens_batch)
        loss = criterion(preds_batch, labels_batch, data_batch)
        running_loss += loss.item()
        n_batches += 1

    return running_loss / n_batches


def train(model, optimizer, criterion, train_ds, val_ds, batch_size, epochs,
          epochs_between_evals=1):
    for epoch in range(1, epochs + 1):

        model.init_hidden(batch_size)

        train_batches = get_batches(train_ds, batch_size)
        for data_batch, labels_batch, lens_batch in train_batches:
            preds_batch = model(data_batch, lens_batch)
            optimizer.zero_grad()
            loss = criterion(preds_batch, labels_batch, data_batch)
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % epochs_between_evals == 0 or epoch == epochs:
            train_loss = evaluate(model, train_ds, criterion, batch_size)
            val_loss = evaluate(model, val_ds, criterion, batch_size)
            print(f'epoch: {epoch:3d}   train_loss: {train_loss:.2f}   val_loss: {val_loss:.2f}')


def generate(model, initial_points, n_points):
    preds = initial_points.unsqueeze(dim=1)

    if torch.cuda.is_available():
        preds = preds.cuda()

    for _ in range(n_points):
        new_preds = model(preds, [len(preds)])
        new_pred = new_preds[-1].unsqueeze(dim=0)
        preds = torch.cat([preds, new_pred])

    preds = preds.int().squeeze().tolist()

    return preds
