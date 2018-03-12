import numpy as np
import os
import ray
import sys
import tensorflow as tf

# Avoids import errors in Ray actor
module_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(module_path, os.pardir))
fusion_path = os.path.join(src_path, "thirdparty/fusion/src2")
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + \
        os.pathsep + fusion_path

sys.path.append(fusion_path)
from checkpointing import variables_to_restore_from_model, initialize_from_checkpoint
from data_providers import make_training_batches, _parse_function
from modeling import model_fn, loss_fn, make_alexnet_featurizer_fn

@ray.remote
class Fusion:
    """Fusion actor.

    Adapted from https://github.com/alexyku/fusion/.
    """
    def __init__(self, alexnet_checkpoint_path, fusion_checkpoint_path):
        """Intializes the fusion actor.

        Args:
            alexnet_checkpoint_path (str): Path to the alexnet checkpoint. For
                example, provide the University of Toronto Alexnet weights
                from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/.
            fusion_checkpoint_path (str): Path to the fusion checkpoint.
        """
        g = tf.Graph()
        with g.as_default():
            self.sess = tf.InteractiveSession(graph=g)

            # TODO: set these from user inputs in case of additional models
            self.inputs_placeholder = tf.placeholder(tf.float32, (None, 4, 227, 227, 35))
            targets_placeholder = tf.placeholder(tf.uint8, (None, 5))

            # Useful for later.
            num_channels = self.inputs_placeholder.shape.as_list()[-1]
            num_classes = targets_placeholder.shape.as_list()[-1]

            # Create the model.
            tf.logging.info("Create the model.")
            # Create AlexNet featurizer function.
            featurizer_fn = make_alexnet_featurizer_fn(
                npy_checkpoint_path=alexnet_checkpoint_path,
                num_channels=num_channels,
                train_alexnet=True)
            # Pass through LSTM model.
            self.logits = model_fn(self.inputs_placeholder, featurizer_fn, num_classes, False)
            init_fn = initialize_from_checkpoint(fusion_checkpoint_path, None, False)
            init_fn(self.sess)

    def get_action(self, sample):
        """Chooses an action given a sample input.

        Args:
            sample (np.array): Sample input to the fusion model. Shape must be
                (4, ?, ?, 35) for current fusion model.

        Returns:
            np.array: one-hot encoded vector representing the predicted action.
        """
        inputs = tf.image.resize_images(sample, size=(227, 227))
        inputs = tf.expand_dims(inputs, 0)
        inputs = inputs.eval()

        pred = self.sess.run(self.logits, feed_dict={self.inputs_placeholder: inputs})
        action = (pred == np.max(pred)).astype(np.uint8)
        return action
