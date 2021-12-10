import torch
import torch.nn as nn
from ..registry import LOSSES


@LOSSES.register_module
class HausdorffLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(HausdorffLoss, self).__init__()
        self.weight = loss_weight

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.ndimension() == 3, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 3, 'got %s' % set2.ndimension()
        if set1.shape[0] > 0:

            assert set1.size()[1] == set2.size()[1], \
                'The points in both sets must have the same number of dimensions, got %s and %s.'\
                % (set1.size()[1], set2.size()[1])

            ''' the second implement to avoid the CUDA out of the memory
                 need the size to be  set1: (N, 40, 2), set2:(N, 40, 2)'''
            d2_matrix = torch.stack([self.cdist(item1, item2) for item1, item2 in zip(set1, set2)], dim=0)
            # Modified Chamfer Loss

            # term_1 = torch.mean(torch.min(d2_matrix, 2)[0].reshape(-1))
            # term_2 = torch.mean(torch.min(d2_matrix, 1)[0].reshape(-1))

            term_1 = torch.mean(torch.min(d2_matrix, 2)[0], -1)
            term_2 = torch.mean(torch.min(d2_matrix, 1)[0], -1)
            res = term_1 + term_2

        else:
            res = set1.sum() * 0
        # print(res)

        return res*self.weight


    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
              i.e. dist[i,j] = ||x[i,:]-y[j,:]||

        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances
