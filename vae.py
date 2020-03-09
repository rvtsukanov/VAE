import torch.nn as nn
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from sklearn.preprocessing import StandardScaler
from metasaver import MetaSaver
import pandas as pd
import matplotlib.pyplot as plt


class VAE(nn.Module, MetaSaver):
    def __init__(
        self,
        num_epochs: int,
        dim_z: int,
        train_set: pd.DataFrame,
        postfix: str,
        base_directory: str,
        lr: float=1e-4,
        render: bool = True,
        render_frequency: int = 10,
    ):
        """
        :param num_epochs: num epochs to evaluate
        :param dim_z: dimentinalion of latent space
        :param train_set: train_set. Necessary to provide full pipleline
        #TODO: use torch.DataLoader to pipeline all transformations
        :param postfix: parameter of MetaSaver: postfix to distinguish experiments
        :param base_directory: root directory to operate
        :param lr: learning rate
        :param render:
        :param render_frequency: num of epochs to plot loss
        """
        super(VAE, self).__init__()
        MetaSaver.__init__(self, base_directory=base_directory, postfix=postfix)

        self.postfix = postfix
        self.scaler = StandardScaler()
        self.scaler.fit(train_set)
        self.train_set_scaled = self.scaler.transform(train_set)
        self.train_df = data_utils.TensorDataset(
            torch.DoubleTensor(self.train_set_scaled), torch.zeros(train_set.shape[0])
        )

        self.train_loader = data_utils.DataLoader(
            self.train_df, batch_size=32, shuffle=True
        )

        feat_shape = self.train_set_scaled.shape[1]

        self.fc_enc_1 = nn.Linear(feat_shape, 32)
        self.fc_enc_2 = nn.Linear(32, 32)
        self.fc_enc_3 = nn.Linear(32, 16)
        self.fc_enc_41 = nn.Linear(16, dim_z)
        self.fc_enc_42 = nn.Linear(16, dim_z)

        self.fc_dec_1 = nn.Linear(dim_z, 16)
        self.fc_dec_2 = nn.Linear(16, 32)
        self.fc_dec_3 = nn.Linear(32, 32)
        self.fc_dec_41 = nn.Linear(32, feat_shape)
        self.fc_dec_42 = nn.Linear(32, feat_shape)
        self.relu = nn.ReLU()

        self.num_epochs = num_epochs
        self.render_frquency = render_frequency
        self.render = render
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def gaussian_sampler(self, mu: torch.Tensor, logvar: torch.Tensor):
        '''
        Reparametrization trick: sample from Gaussian
        :param mu:
        :param logvar:
        :return:
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        '''
        Encoder of VAE
        :param x: input data
        :return: vector in latent space
        '''
        h1 = self.relu(self.fc_enc_1(x))
        h2 = self.relu(self.fc_enc_2(h1))
        h3 = self.relu(self.fc_enc_3(h2))
        return self.fc_enc_41(h3), torch.sigmoid(self.fc_enc_42(h3))

    def decode(self, z: torch.Tensor):
        '''
        Decoder of VAE
        :param z: sampled vector from parametrized latent space
        :return: vector in initial space
        '''
        h1 = self.relu(self.fc_dec_1(z))
        h2 = self.relu(self.fc_dec_2(h1))
        h3 = self.relu(self.fc_dec_3(h2))
        return self.fc_dec_41(h3), self.fc_dec_42(h3)

    def forward(self, x: torch.Tensor):
        latent_mu, latent_logsigma = self.encode(x)
        z = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(z)
        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma

    def kl(self, logsigma: torch.Tensor, mu: torch.Tensor):
        '''
        KL-divergence between to Gaussians.
        https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
        :return: value of divergence
        '''
        return -0.5 * torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma))

    def reconstruction_loss(self, mu_gen: torch.Tensor,
                            x: torch.Tensor):
        '''
        Measure of quality of reconstruction. Here: MSE between original and reconstructed data
        :param mu_gen: vector of generated points
        :param x: original data
        :return: value of loss
        '''
        return torch.mean(torch.sum((mu_gen - x) ** 2))

    def loss_vae(self, x: torch.Tensor,
                 mu_gen: torch.Tensor,
                 mu_z: torch.Tensor,
                 logsigma_z: torch.Tensor):
        '''
        Total loss = KL + RECONSTRUCTION
        :param x: original data
        :param mu_gen: vector of generated points
        :param mu_z: vector of means in latent space
        :param logsigma_z: vector of log(std) in latent space
        :return:
        '''
        return self.kl(mu_z, logsigma_z) + self.reconstruction_loss(mu_gen, x)

    def set_postfix(self, new_postfix: str):
        self.postfix = new_postfix

    def run(self):
        '''
        Main loop of learning
        '''
        self.train_loss_epoch = []
        for epoch in range(self.num_epochs):
            self.train()
            train_loss = []

            for obj, y in self.train_loader:
                self.optimizer.zero_grad()
                mu_reconsct, logsigma_reconsct, mu_noise, logsigma_noise = self(
                    obj.float()
                )
                loss = self.loss_vae(
                    obj, mu_reconsct, mu_noise, logsigma_noise
                )
                loss.backward()
                self.optimizer.step()
                self.logger_inner.info(loss)
                train_loss.append(loss.data.numpy())

            self.train_loss_epoch.append(np.mean(train_loss))
            self.writer.add_scalar('Reconstruction loss', np.mean(train_loss), epoch)
            if self.render and epoch % self.render_frquency == 0:
                plt.plot(np.array(self.train_loss_epoch))
                plt.show()
        return self

    def generate_from_noise(self, batch_of_input_data: pd.DataFrame,
                            num_of_sampled: int):
        '''
        1. Computing latent alignment for points in batch_of_input_data P(z | X)
        2. Sampling "num_of_sampled" points from P(z | X)
        3. Reconstructing them by pipeline
        :return: reconstructed num_of_sampled points in original space,
        reconstructed num_of_sampled points in normalized space
        '''
        input_data_scaled = torch.Tensor(self.scaler.transform(batch_of_input_data))
        latent_mu, latent_logsigma = self.encode(input_data_scaled)

        dist = Normal(loc=latent_mu, scale=latent_logsigma.exp())
        sampled = dist.sample(torch.Size([num_of_sampled]))

        return (
            self.scaler.inverse_transform(self.decode(sampled)[0].detach().numpy()),
            self.decode(sampled),
        )
