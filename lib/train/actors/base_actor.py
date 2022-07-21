from lib.utils import TensorDict


class BaseActor:
    """
    Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss.
    """

    def __init__(self, net, objective):
        """
        Args:
            net: The network to train.
            objective: The loss function.
        """

        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """
        Called in each training iteration. Should pass in input data through the network,
        calculate the loss, and return the training stats for the input data.

        Args:
            data: A TensorDict containing all the necessary data blocks.

        Returns:
            loss: Loss for the input data.
            stats: A dict containing detailed losses.
        """

        raise NotImplementedError

    def to(self, device):
        """
        Move the network to device.

        Args:
            device: Device to use ('cpu' or 'cuda').
        """

        self.net.to(device)

    def train(self, mode=True):
        """
        Set whether the network is in train mode.

        Args:
            mode (True): Bool specifying whether in training mode.
        """

        self.net.train(mode)

    def eval(self):
        """
        Set network to eval mode.
        """

        self.train(False)
