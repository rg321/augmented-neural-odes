import torch
import torch.nn as nn
from anode.models import ODEBlock
from torchdiffeq import odeint, odeint_adjoint
import torch.nn.functional as F


def norm(dim, type='group_norm'):
    if type == 'group_norm':
        return nn.GroupNorm(dim, dim)
    return nn.BatchNorm2d(dim)

class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)


class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, img_size, num_filters, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ConvODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.img_size = img_size
        self.time_dependent = time_dependent
        self.nfe = 0  # Number of function evaluations
        self.channels, self.height, self.width = img_size
        self.channels=64
        # print('image size ......', img_size)
        # print('num channels ........ ', self.channels)
        self.channels += augment_dim
        # print('num channels ........ final ......... ', self.channels)
        self.num_filters = num_filters
        # self.norm = nn.BatchNorm2d(num_filters)

        if time_dependent:
            self.conv1 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv3 = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:

            self.conv1 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        self.nfe += 1
        if self.time_dependent:
            out = self.conv1(t, x)
            out = self.non_linearity(out)
            out = self.conv2(t, out)
            out = self.non_linearity(out)
            out = self.conv3(t, out)
        else:
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.augment_dim,
                              height, width).to(self.device)
            x_aug = torch.cat([x, aug], 1)
            out = self.conv1(x_aug)
            out = self.norm(out)
            out = self.non_linearity(out)
            out = self.conv2(out)
            out = self.norm(out)
            out = self.non_linearity(out)
            out = self.conv3(out)
            out = self.norm(out)
        return out


class ConvODENet(nn.Module):
    """Creates an ODEBlock with a convolutional ODEFunc followed by a Linear
    layer.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, img_size, num_filters, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ConvODENet, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_filters = num_filters
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.flattened_dim = (img_size[0] + augment_dim) * img_size[1] * img_size[2]
        self.time_dependent = time_dependent
        self.tol = tol
        self.total_dim = 64 + self.augment_dim
        self.downsampling = nn.Sequential(
            nn.Conv2d(img_size[0], 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )

        odefunc = ConvODEFunc(device, img_size, num_filters, augment_dim,
                              time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, is_conv=True, tol=tol,
                                 adjoint=adjoint)

        # self.linear_layer = nn.Linear(self.flattened_dim, self.output_dim)
        self.fc = nn.Linear(self.total_dim, self.output_dim)

    def forward(self, x, return_features=False):
        # features = self.odeblock(x)
        # pred = self.linear_layer(features.view(features.size(0), -1))
        # if return_features:
        #     return features, pred
        # return pred
        x = self.downsampling(x)
        batch_size, channels, height, width = x.shape
        # print(x.shape)
        # aug = torch.zeros(batch_size, augment_dim,
        #                   height, width).to(self.device)
        # x_aug = torch.cat([x, aug], 1)
        # x = self.downsampling(x)
        # x = self.feature(x)
        x = self.odeblock(x)
        # batch x 64 x 6 x 6
        total_dim = 64 + self.augment_dim
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        x = norm(total_dim)(x)
        x = F.relu(x)
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out
