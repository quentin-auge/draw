import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.dataset import PADDING_VALUE
from lib.dataset import get_batches


class EncoderDecoder(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.encoder = Encoder(n_hidden)
        self.decoder = Decoder(n_hidden)

    def forward(self, data, lens):
        states = self.encoder(data, lens)
        output, _ = self.decoder(data, lens, encoder_states=states)
        return output


class Encoder(nn.Module):
    def __init__(self, n_hidden, n_layers=1):
        super().__init__()

        self.n_layers = n_layers

        self.lstm = nn.LSTM(5, n_hidden, n_layers, bidirectional=True)

    def forward(self, data, lens):
        packed_data = pack_padded_sequence(data, lens)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_data)
        output, _ = pad_packed_sequence(packed_output, padding_value=PADDING_VALUE)
        hidden_state = torch.cat([hidden_state[0], hidden_state[1]], dim=-1).unsqueeze(dim=0)
        cell_state = torch.cat([cell_state[0], cell_state[1]], dim=-1).unsqueeze(dim=0)
        return hidden_state, cell_state


class Decoder(nn.Module):
    def __init__(self, n_hidden, n_layers=1, K=6):
        super().__init__()

        self.n_layers = n_layers
        self.K = K

        #self.hidden_bridge = nn.Linear(2 * n_hidden, n_hidden)
        #self.cell_bridge = nn.Linear(2 * n_hidden, n_hidden)
        self.lstm = nn.LSTM(5, n_hidden, n_layers)
        self.output_weights = nn.Linear(n_hidden, 6 * K + 3)

    def forward(self, data, lens, states=None):

        #if not states and encoder_states:
        #    hidden_state = torch.tanh(self.hidden_bridge(encoder_states[0]))
        #    cell_state = torch.tanh(self.cell_bridge(encoder_states[1]))
        #    states = hidden_state, cell_state

        packed_data = pack_padded_sequence(data, lens)
        packed_output, states = self.lstm(packed_data, states)
        output, _ = pad_packed_sequence(packed_output, padding_value=PADDING_VALUE)

        #output, states = self.lstm(data, states)

        params = self.output_weights(output)
        params = params.split(6, -1)

        state_params = params[-1]
        state_params = state_params.softmax(dim=-1)

        gmm_params = torch.cat(params[:-1], dim=-1)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = gmm_params.split(self.K, dim=-1)

        pi = pi.softmax(dim=-1)
        sigma_x = sigma_x.exp()
        sigma_y = sigma_y.exp()
        rho_xy = rho_xy.tanh()

        mu_x = mu_x.unsqueeze(-1)
        mu_y = mu_y.unsqueeze(-1)
        sigma_x = sigma_x.unsqueeze(-1)
        sigma_y = sigma_y.unsqueeze(-1)
        rho_xy = rho_xy.unsqueeze(-1)

        mu = torch.cat([mu_x, mu_y], dim=-1)

        sigma_xy = sigma_x * sigma_y
        upper_row_cov = torch.cat([sigma_x ** 2, rho_xy * sigma_xy], dim=-1).unsqueeze(-2)
        lower_row_cov = torch.cat([rho_xy * sigma_xy, sigma_y ** 2], dim=-1).unsqueeze(-2)
        cov = torch.cat([upper_row_cov, lower_row_cov], dim=-2)

        return (pi, mu, cov), state_params, states


# def get_reg_loss(preds, labels):
#     n_preds = len(preds)
#     labels = labels[:n_preds, :, :]
#
#     masked_labels = labels[labels != PADDING_VALUE]
#     masked_preds = preds[labels != PADDING_VALUE]
#
#     return F.mse_loss(masked_preds, masked_labels)
#
#
# def get_classif_loss(preds, labels):
#     n_preds = len(preds)
#     labels = labels[:n_preds, :, :]
#
#     masked_labels = labels[labels != PADDING_VALUE]
#     masked_preds = preds[labels != PADDING_VALUE]
#
#     return F.mse_loss(masked_preds, masked_labels)
#
#     #return (1 - masked_preds[masked_labels == 1]).mean()
#
#
# def get_loss(preds, labels):
#     reg_loss = get_reg_loss(preds[:, :, :2], labels[:, :, :2])
#     classif_loss = get_classif_loss(preds[:, :, 2:], labels[:, :, 2:])
#     return (reg_loss, classif_loss)


def evaluate(model, criterion, ds, mean_stds, batch_size=1024):
    running_loss = 0
    n_batches = 0

    batches = get_batches(ds, mean_stds, batch_size)
    for data_batch, labels_batch, lens_batch in batches:
        (pi, mu, cov), param_states, _ = model(data_batch, lens_batch)
        loss = criterion(pi, mu, cov, param_states, labels_batch, lens_batch)
        running_loss += loss.item()
        n_batches += 1

    loss = running_loss / n_batches

    return loss


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
            (pi, mu, cov), param_states, _ = model(data_batch, lens_batch)
            optimizer.zero_grad()
            loss = criterion(pi, mu, cov, param_states, labels_batch, lens_batch)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, criterion, train_ds, train_means_stds, batch_size)
        val_loss = evaluate(model, criterion, val_ds, val_means_stds, batch_size)

        if scheduler:
            scheduler.step(val_loss)

        if epoch == 1 or epoch % epochs_between_evals == 0 or epoch == epochs:

            print(f'epoch: {epoch:3d}'
                  f'   train_loss: {train_loss:.5f}'
                  f'   val_loss: {val_loss:.5f}')


# def generate(model, start_of_stroke, n_points):
#
#     decoder = model.decoder
#     encoder = model.encoder
#
#     preds = start_of_stroke
#     encoder_states = encoder(start_of_stroke, [len(start_of_stroke)])
#
#     for _ in range(n_points):
#         new_preds, states = decoder(preds, [len(preds)], encoder_states=encoder_states)
#         new_pred = new_preds[-1].unsqueeze(dim=0)
#         preds = torch.cat([preds, new_pred])
#
#     preds = preds.squeeze()
#
#     return preds
