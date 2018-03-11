import numpy as np
import os
import ray
import sys
import tensorflow as tf

module_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(module_path, os.pardir))
models_research_path = os.path.join(src_path, "thirdparty/models/research")

# Avoids import errors in Ray actor
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") \
        + os.pathsep + models_research_path

sys.path.append(models_research_path)
from thirdparty.models.research.object_detection.utils \
        import label_map_util   # noqa


# TODO: convert these to arguments for ObjectDetector
CUTOFF_PERCENTAGE = 0.5
NUM_ENCODING_CLASSES = 13
CATEGORY_INDEX_TO_ENCODING = {
        1: 0,       # person -> person
        2: 1,       # bicycle -> bicycle
        3: 2,       # car -> car
        4: 3,       # motorcycle -> motorcycle
        6: 4,       # bus -> bus
        7: 5,       # train -> train
        8: 6,       # truck -> truck
        9: 7,       # boat -> boat
        10: 8,      # traffic light -> traffic light
        11: 9,      # fire hydrant -> fire hydrant
        13: 10,     # stop sign -> stop sign
        14: 11,     # parking meter -> parking meter
        16: 12,     # bird -> animal
        17: 12,     # cat -> animal
        18: 12,     # dog -> animal
        19: 12,     # horse -> animal
        20: 12,     # sheep -> animal
        21: 12,     # cow -> animal
        22: 12,     # elephant -> animal
        23: 12,     # bear -> animal
        24: 12,     # zebra -> animal
        25: 12,     # giraffe -> animal
        }


@ray.remote
class ObjectDetector:
    """Object detection actor.

    Adapted from https://github.com/tensorflow/models.
    """
    def __init__(self, path_to_ckpt, path_to_labels, num_classes):
        """Initializes the object detection actor.

        Args:
            path_to_ckpt (str): Path to the object detection checkpoint.
            path_to_labels (str): Path to file containing the list of objects
                the model can detect.
            num_classes (int): Maximum number of objects the model can detect.
        """
        # Load model
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        # Load label map
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
                        label_map, max_num_classes=num_classes,
                        use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # Initialize tensors with detection_graph.as_default():
        self.sess = tf.Session(graph=self.detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name(
                                "image_tensor:0")
        # Each box represents a part of the image where a particular object was
        # detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name(
                                    "detection_boxes:0")
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name(
                                    "detection_scores:0")
        self.detection_classes = self.detection_graph.get_tensor_by_name(
                                    "detection_classes:0")
        self.num_detections = self.detection_graph.get_tensor_by_name(
                                    "num_detections:0")

    def detect_objects(self, image):
        """Detect objects in the provided image.

        Args:
            image (np.array): 3 channel image.

        Returns:
            np.array: n channel image where n is the number of encoding
            classes used.
        """
        image = np.copy(image)
        # Expand dimensions since the model expects images to have shape:
        # [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
          [self.detection_boxes, self.detection_scores, self.detection_classes,
           self.num_detections],
          feed_dict={self.image_tensor: image_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.uint8)
        scores = np.squeeze(scores)
        # Create one-hot encodings
        encodings = np.zeros((image.shape[0], image.shape[1],
                              NUM_ENCODING_CLASSES), dtype=np.uint8)
        for box, cls, score in zip(boxes, classes, scores):
            if score < CUTOFF_PERCENTAGE:
                break
            ymin = int(round(box[0] * image.shape[0]))
            ymax = int(round(box[2] * image.shape[0]))
            xmin = int(round(box[1] * image.shape[1]))
            xmax = int(round(box[3] * image.shape[1]))

            idx = CATEGORY_INDEX_TO_ENCODING[cls]
            if idx is not None:
                # Use max + 1 in indexing to include max pixel
                encodings[ymin:ymax + 1, xmin:xmax + 1, idx] = 1

        return encodings
