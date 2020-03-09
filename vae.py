import torch.nn as nn
import torch
import torch.utils.data as data_utils
import torch.optim as optim
import numpy as np
import os
from torch.distributions.normal import Normal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from metasaver import MetaSaver


class VAE(nn.Module, MetaSaver):
    def __init__(self, num_epochs, dim_z, train_set, postfix, base_directory, lr=1e-4):
        super(VAE, self).__init__()
        MetaSaver.__init__(self, base_directory=base_directory,
                           postfix=postfix)

        self.postfix = postfix

        self.scaler = StandardScaler()
        self.scaler.fit(train_set)

        self.train_set_scaled = self.scaler.transform(train_set)
        self.train_df = data_utils.TensorDataset(torch.DoubleTensor(self.train_set_scaled),
                                                 torch.zeros(train_set.shape[0]))

        self.train_loader = data_utils.DataLoader(self.train_df, batch_size=32, shuffle=True)

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
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=lr)


    def gaussian_sampler(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h1 = self.relu(self.fc_enc_1(x))
        h2 = self.relu(self.fc_enc_2(h1))
        h3 = self.relu(self.fc_enc_3(h2))
        return self.fc_enc_41(h3), torch.sigmoid(self.fc_enc_42(h3))

    def decode(self, z):
        h1 = self.relu(self.fc_dec_1(z))
        h2 = self.relu(self.fc_dec_2(h1))
        h3 = self.relu(self.fc_dec_3(h2))
        return self.fc_dec_41(h3), self.fc_dec_42(h3)

    def forward(self, x):
        '''
        :param x:
        :return:
        '''
        latent_mu, latent_logsigma = self.encode(x)
        z = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(z)
        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma


    def kl(self, logsigma, mu):
        return -0.5 * torch.sum(1 + logsigma - mu ** 2 - torch.exp(logsigma))


    def reconstruction_loss(self, mu_gen, x):
        return torch.mean(torch.sum((mu_gen - x) ** 2))


    def loss_vae(self, x, mu_gen, logsigma_gen, mu_z, logsigma_z):
        return self.kl(mu_z, logsigma_z) + self.reconstruction_loss(mu_gen, x)

    def set_postfix(self, new_postfix):
        self.postfix = new_postfix

    def run(self):
        self.train_loss_epoch = []
        for epoch in range(self.num_epochs):
            # self.train() #TODO: rename
            train_loss = []

            for obj, y in self.train_loader:

                self.optimizer.zero_grad()
                mu_reconsct, logsigma_reconsct, mu_noise, logsigma_noise = self(obj.float())

                loss = self.loss_vae(obj, mu_reconsct, logsigma_reconsct, mu_noise, logsigma_noise)
                # print(loss)

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.data.numpy())
            self.train_loss_epoch.append(np.mean(train_loss))
            if epoch % 10 == 0:
                pass
                # plt.plot(np.array(self.train_loss_epoch))
                # plt.show()
        return self


    def save_loss(self):
        if not os.path.exists(f'{self.name_dir}'):
            os.makedirs(f'./{self.name_dir}')
        np.save(open(f'{self.name_dir}/loss.npy', 'wb'), np.array(self.train_loss_epoch))
        return self


    def save_model(self):
        if not os.path.exists(f'{self.name_dir}'):
            os.makedirs(f'./{self.name_dir}')
        torch.save(self.state_dict(), f'{self.name_dir}/model')
        return self



    def load_loss(self, dir):
        np.load(open(os.path.join(dir, 'loss.npy')))


    def plot_loss(self, dir=None):
        pass


    def render_configuration_into_file(self):
        pass #TODO


    def generate_from_noise(self, batch_of_input_data, num_of_sampled):
        input_data_scaled = torch.Tensor(self.scaler.transform(batch_of_input_data))
        # print(input_data_scaled)
        latent_mu, latent_logsigma = self.encode(input_data_scaled)

        dist = Normal(loc=latent_mu, scale=latent_logsigma.exp())
        sampled = dist.sample(torch.Size([num_of_sampled]))

        # sampled = sampled.transpose(1, 0)

        # print(self.decode(sampled)[0].shape)
        # print(self.decode(sampled))


        return self.scaler.inverse_transform(self.decode(sampled)[0].detach().numpy()), self.decode(sampled)





