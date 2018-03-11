import os
import numpy as np
import ray
import sys
import torch
from torch.autograd import Variable

# Avoids import errors in Ray actor
module_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(module_path, os.pardir))
drn_path = os.path.join(src_path, "thirdparty/drn")
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + \
        os.pathsep + drn_path

sys.path.append(drn_path)
from segment import DRNSeg # noqa


@ray.remote(num_gpus=1)
class Segmentor:
    """Segmentation actor.

    Adapted from https://github.com/fyu/drn.
    """
    def __init__(self, arch, classes, pretrained):
        """Initializes the segmentation actor.

        Args:
            arch (str): Architecture of the segmentation neural network.
                Options include: drn-c-26, drn-c-42, drn-c-58, drn-d-22,
                drn-d-38, drn-d-54, drn-d-105
            classes (int): The number of classes identified by segmentation.
            pretrained (str): Path to the pretrained model.

        Todo:
            Improve variable names.
            Load from model zoo if no local model specified (see drn/drn.py).
        """
        single_model = DRNSeg(arch, classes, pretrained_model=None,
                              pretrained=False)
        single_model.load_state_dict(torch.load(pretrained))
        self.model = torch.nn.DataParallel(single_model).cuda()
        self.num_classes = classes

    def segment_image(self, image):
        """Creates a one-hot encoded image segmentation of the current image.

        Args:
            image (np.array): Image containing RGB channels.

        Returns:
            np.array: One-hot encoded array with identical dimensions to image
                and number of channels equal to `classes`.
        """
        # Load image from numpy array to pytorch variable
        image = torch.from_numpy(image.transpose([2, 0, 1])) \
            .unsqueeze(0).float()
        image_var = Variable(image, requires_grad=False, volatile=True)

        # Get model prediction
        final = self.model(image_var)[0]
        _, pred = torch.max(final, 1)

        segmentation = pred.cpu().data.numpy()[0].astype(np.uint8)

        one_hot = np.zeros((segmentation.shape[0], segmentation.shape[1],
                            self.num_classes), dtype=np.uint8)
        # Consider moving this out of the actor to reduce object store storage
        for i in range(self.num_classes):
            one_hot[:, :, i] = (segmentation == i)

        return one_hot
