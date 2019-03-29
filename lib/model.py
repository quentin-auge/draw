from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions.multinomial import Multinomial
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.dataset import get_batches


class Encoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_latent, bidirectional=True):
        super().__init__()

        self.dim_latent = dim_latent

        self.lstm = nn.LSTM(dim_input, dim_hidden, bidirectional=bidirectional)

        dim_hidden_out = dim_hidden * 2 if bidirectional else dim_hidden
        self.mu = nn.Linear(dim_hidden_out, dim_latent)
        self.sigma = nn.Linear(dim_hidden_out, dim_latent)

    def forward(self, data, lengths):
        packed_data = pack_padded_sequence(data, lengths)
        packed_output, (hidden_state, _) = self.lstm(packed_data)
        output, _ = pad_packed_sequence(packed_output, padding_value=0)

        # Split forward and backward hidden states, each of shape 1 * batch_size * dim_hidden
        fwd_hidden_state, bwd_hidden_state = torch.split(hidden_state, 1, dim=0)
        # Put them back as tensor of shape 1 * batch_size * (2 * dim_hidden)
        hidden_state = torch.cat([fwd_hidden_state, bwd_hidden_state], dim=-1)

        mu = self.mu(hidden_state)
        sigma_hat = self.sigma(hidden_state)
        sigma = (sigma_hat / 2).exp()

        z = torch.normal(mu, sigma)

        return z, mu, sigma_hat


class Decoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_gaussians):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.n_gaussians = n_gaussians

        self.lstm = nn.LSTM(dim_input, dim_hidden)
        self.output_weights = nn.Linear(dim_hidden, 6 * n_gaussians + 3)

    def forward(self, data, lengths, lstm_states=None, temperature_gmm=1.0, temperature_state=1.0):
        packed_data = pack_padded_sequence(data, lengths)
        packed_output, lstm_states = self.lstm(packed_data, lstm_states)
        output, _ = pad_packed_sequence(packed_output, padding_value=0)

        all_params = self.output_weights(output)
        all_params = all_params.split(6, -1)

        # Shape of strokes_state_params: max_sequence_length_in_batch * batch_size * 3
        strokes_state_params = all_params[-1]
        strokes_state_params = (strokes_state_params / temperature_state).softmax(dim=-1)

        gmm_params = torch.cat(all_params[:-1], dim=-1)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = gmm_params.split(self.n_gaussians, dim=-1)

        pi = (pi / temperature_gmm).softmax(dim=-1)
        sigma_x = sigma_x.exp() * np.sqrt(temperature_gmm)
        sigma_y = sigma_y.exp() * np.sqrt(temperature_gmm)
        rho_xy = rho_xy.tanh()

        # Shape of each of gmm_params: max_sequence_length_in_batch * batch_size * n_gaussians
        gmm_params = pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy

        return gmm_params, strokes_state_params, lstm_states


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.bridge = nn.Linear(encoder.dim_latent, 2 * decoder.dim_hidden)
        self.decoder = decoder

    def forward(self, data, lengths):
        encoder_out = self.encoder(data, lengths)
        z, _, _ = encoder_out
        decoder_data = self.get_decoder_data(data, z)
        decoder_states = self.get_decoder_states(z)
        decoder_out = self.decoder(decoder_data, lengths, decoder_states)
        return decoder_out

    def get_decoder_data(self, data, z):
        # z: 1 * batch_size * n_latent
        # z_repeated: max_sequence_length_in_dataset * batch_size * n_latent
        z_repeated = z.repeat(len(data), 1, 1)

        # data: max_sequence_length_in_dataset * batch_size * n_latent
        # decoder_data: max_sequence_length_in_dataset * batch_size * (n_latent + 5)
        decoder_data = torch.cat([data, z_repeated], dim=-1)

        return decoder_data

    def get_decoder_states(self, z):
        decoder_states = torch.tanh(self.bridge(z))
        decoder_states = decoder_states.split(self.decoder.dim_hidden, dim=-1)
        return decoder_states


def reconstruction_loss(gmm_params, strokes_state_params, labels_batch, lengths_batch):
    pi = gmm_params[0]
    n_gaussians = pi.shape[2]

    # Create lengths mask for predictions
    max_length = lengths_batch.max()
    batch_size = labels_batch.shape[1]
    mask = torch.zeros(max_length, batch_size)
    for i, length in enumerate(lengths_batch):
        mask[:length, i] = 1

    # Shape of labels_batch: max_sequence_length_in_dataset * batch_size * 5
    # Shape of stripped_labels_batch: max_sequence_length_in_batch * batch_size * 5
    stripped_labels_batch = labels_batch[:max_length, :, :]

    # Compute GMM loss

    trajectory = stripped_labels_batch[:, :, :2]

    # Add extra rank of size n_gaussians
    trajectory_x, trajectory_y = trajectory.split(1, dim=-1)
    trajectory_x = trajectory_x.repeat(1, 1, n_gaussians)
    trajectory_y = trajectory_y.repeat(1, 1, n_gaussians)

    # Do not use torch...MultivariateNormal for loss. It is way too slow on GPU.
    gaussian_probs = bivariate_normal_pdf(trajectory_x, trajectory_y, *gmm_params[1:])

    gmm_probas = (gaussian_probs * pi).sum(dim=-1)
    # Prevent zeroes from showing up in log
    gmm_probas += 1e-5
    gmm_loss = -gmm_probas[mask == 1].log().mean()

    # Compute strokes state loss

    strokes_state = stripped_labels_batch[:, :, 2:]
    states_losses = (strokes_state * strokes_state_params.log()).sum(dim=-1)
    strokes_states_loss = -states_losses[mask == 1].mean()

    # Full loss

    return gmm_loss + strokes_states_loss


def bivariate_normal_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    z_x = ((x - mu_x) / sigma_x) ** 2
    z_y = ((y - mu_y) / sigma_y) ** 2
    z_xy = (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y)
    z = z_x + z_y - 2 * rho_xy * z_xy
    exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))
    norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)

    return exp / norm


def generate(model, n_points, initial_points=None, temperature_gmm=1.0, temperature_state=1.0):
    if isinstance(model, EncoderDecoder):
        # Conditional generation
        z, _, _ = model.encoder(initial_points, [len(initial_points)])
        decoder = model.decoder
        get_decoder_data_func = partial(model.get_decoder_data, z=z)
        decoder_states = model.get_decoder_states(z)
    else:
        # Unconditional generation
        decoder = model
        get_decoder_data_func = lambda data: data
        decoder_states = None

    initial_points = torch.Tensor([[[0, 0, 1, 0, 0]]]).float()

    if torch.cuda.is_available():
        initial_points = initial_points.cuda()

    point = initial_points
    preds = [initial_points]

    for _ in range(n_points):
        decoder_data = get_decoder_data_func(point)

        with torch.no_grad():
            out = decoder(decoder_data, [len(decoder_data)], decoder_states,
                          temperature_gmm, temperature_state)
            gmm_params, strokes_state_params, decoder_states = out
            last_gmm_params = [param[-1] for param in gmm_params]

        point = sample(last_gmm_params, strokes_state_params)
        point = point.unsqueeze(0)
        preds.append(point)

    preds = torch.cat(preds).squeeze(1)

    return preds


def sample(gmm_params, strokes_state_params):
    # Shape of each of gmm_params: batch_size * n_gaussians (output of LSTM for last sequence
    # element)
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = gmm_params

    gaussians = bivariate_normal_distribution(mu_x, mu_y, sigma_x, sigma_y, rho_xy)
    # Shape of gaussians_preds: batch_size * n_gaussians * 2
    gaussians_preds = gaussians.sample()
    # Shape of pi before: batch_size * n_gaussians
    # Shape of pi after: batch_size * n_gaussians * 2
    pi = pi.unsqueeze(dim=-1).repeat(1, 1, 2)
    # Shape of gmm_preds: batch_size * 2
    gmm_preds = (pi * gaussians_preds).sum(dim=-2)

    # Shape of strokes_state_preds: batch_size * 3
    strokes_state_preds = Multinomial(1, strokes_state_params[-1]).sample()

    # Shape of preds: batch_size * 5
    preds = torch.cat([gmm_preds, strokes_state_preds], dim=1)

    return preds


def bivariate_normal_distribution(mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    # Shape of each of gmm_params: max_sequence_length_in_batch * batch_size * n_gaussians
    mu_x = mu_x.unsqueeze(-1)
    mu_y = mu_y.unsqueeze(-1)
    sigma_x = sigma_x.unsqueeze(-1)
    sigma_y = sigma_y.unsqueeze(-1)
    rho_xy = rho_xy.unsqueeze(-1)

    # Shape of mu: max_sequence_length_in_batch * batch_size * n_gaussians * 2
    mu = torch.cat([mu_x, mu_y], dim=-1)

    sigma_xy = sigma_x * sigma_y
    upper_row_cov = torch.cat([sigma_x ** 2, rho_xy * sigma_xy], dim=-1).unsqueeze(-2)
    lower_row_cov = torch.cat([rho_xy * sigma_xy, sigma_y ** 2], dim=-1).unsqueeze(-2)
    # Shape of cov: max_sequence_length_in_batch * batch_size * n_gaussians * 2 * 2
    cov = torch.cat([upper_row_cov, lower_row_cov], dim=-2)

    return MultivariateNormal(mu, cov)


def train(model, scheduler_or_optimizer, criterion, train_ds, val_ds,
          train_stds, val_stds, batch_size, epochs, epochs_between_evals=1):
    if isinstance(scheduler_or_optimizer, torch.optim.Optimizer):
        optimizer = scheduler_or_optimizer
        scheduler = None
    else:
        scheduler = scheduler_or_optimizer
        optimizer = scheduler.optimizer

    for epoch in range(1, epochs + 1):

        train_batches = get_batches(train_ds, train_stds, batch_size)
        for data_batch, labels_batch, lengths_batch in train_batches:
            gmm_params, param_states, _ = model(data_batch, lengths_batch)
            optimizer.zero_grad()
            loss = criterion(gmm_params, param_states, labels_batch, lengths_batch)
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, criterion, train_ds, train_stds, batch_size)
        val_loss = evaluate(model, criterion, val_ds, val_stds, batch_size)

        if scheduler:
            scheduler.step(val_loss)

        if epoch == 1 or epoch % epochs_between_evals == 0 or epoch == epochs:
            print(f'epoch: {epoch:3d}'
                  f'   train_loss: {train_loss:.5f}'
                  f'   val_loss: {val_loss:.5f}')


def evaluate(model, criterion, ds, stds, batch_size=1024):
    running_loss = 0
    n_batches = 0

    batches = get_batches(ds, stds, batch_size)
    for data_batch, labels_batch, lengths_batch in batches:
        gmm_params, param_states, _ = model(data_batch, lengths_batch)
        loss = criterion(gmm_params, param_states, labels_batch, lengths_batch)
        running_loss += loss.item()
        n_batches += 1

    loss = running_loss / n_batches

    return loss
