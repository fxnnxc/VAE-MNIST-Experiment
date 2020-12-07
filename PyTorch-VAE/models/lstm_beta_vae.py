import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        #output = embedded
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size) #device= device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        #self.out = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #output = self.embedding(input).view(1, 1, -1)
        #output = F.relu(output)
        output, hidden = self.gru(input, hidden)
        #output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)# device=input.device)



class LSTMBetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 input_size = 32,
                 **kwargs) -> None:
        super(LSTMBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.use_teacher_forcing = True
        self.input_size = input_size

        if hidden_dims is None:
            hidden_dims = [10]

        self.encoder = EncoderRNN(input_size, input_size)
        self.fc_mu = nn.Linear(input_size, latent_dim)
        self.fc_var = nn.Linear(input_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, input_size)
        self.decoder = DecoderRNN(input_size, input_size)


    def encode(self, input: Tensor) -> List[Tensor]:

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(self.input_size, self.encoder.hidden_size)

        for ei in range(self.input_size):
            print(input[ei].size())
            encoder_output, encoder_hidden = self.encoder(input[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        result = encoder_hidden

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, input) -> Tensor:

        result = self.decoder_input(z)
        decoder_input = torch.tensor([[0 for i in range(self.input_size)]])
        decoder_hidden = result

        if self.use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(self.input_size):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
                decoder_input = input[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(self.input_size):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        print(args[0].size())
        print(args[1].size())
        print(args[2].size())
        print(args[3].size())


        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]