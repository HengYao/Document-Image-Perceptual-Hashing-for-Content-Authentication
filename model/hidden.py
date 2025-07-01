import numpy as np
import torch
import torch.nn as nn
from ImageHashModel import ResNetVit
from imagetext import FusionModel
from options import HiDDenConfiguration

class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device):

        super(Hidden, self).__init__()
        self.deephash = FusionModel(configuration,hidden_dim=256, n_class=50,image_dim=256).to(device)
        self.optimizer = torch.optim.Adam(self.deephash.parameters(),lr=0.0001)
        self.config = configuration
        self.device = device
        self.mse_loss = nn.MSELoss().to(device)


    def train_on_batch(self, tokens, segments, input_masks, image):

        batch_size = image.shape[0]

        with torch.enable_grad():

            hash_code = self.deephash(tokens, segments,input_masks,image)

            loss_sim_sum = 0
            loss_dif_sum = 0

            self.optimizer.zero_grad()

            for num_in_fea in range(1, 36):

                loss_sim = self.mse_loss(hash_code[0], hash_code[num_in_fea])

                loss_sim = torch.sigmoid(loss_sim)


                loss_sim_sum = loss_sim_sum + loss_sim

            for num_in_fea1 in range(36,71):
                loss_dif = self.mse_loss(hash_code[0], hash_code[num_in_fea1])

                loss_dif = torch.sigmoid(loss_dif)


                loss_dif_sum = loss_dif_sum + loss_dif


            loss = (loss_sim_sum/35) - (loss_dif_sum/35)



            loss.backward(retain_graph=True)
            loss.backward()


            self.optimizer.step()

        losses = {
            'similar_loss    ': (loss_sim_sum/35).item(),
            'different_loss  ': (loss_dif_sum/35).item(),
            'loss            ': loss.item()
        }

        return losses, hash_code


    def validate_on_batch(self,tokens, segments, input_masks, image):

        batch_size = image.shape[0]
        loss = []
        with torch.no_grad():

            hash_code = self.deephash(tokens, segments,input_masks,image)
            for num_in_fea in range(1, batch_size):
                loss.append(self.mse_loss(hash_code[0], hash_code[num_in_fea]))

        return loss, hash_code

    def test_single(self,tokens,segments,input_masks,image):

        with torch.no_grad():

            hash_code = self.deephash(tokens,segments,input_masks,image)

        return hash_code

    def to_stirng(self):
        return '{}\n'.format(str(self.deephash))
