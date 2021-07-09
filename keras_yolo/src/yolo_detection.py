import colorsys
import os
from timeit import default_timer as timer
import tensorflow
import numpy as np
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import tensorflow.compat.v1.keras.backend as K
tensorflow.compat.v1.disable_eager_execution()
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw

from yolov3 import yolo_eval, yolo_body, tiny_yolo_body
from utils import letterbox_image

class YOLO(object):
    _defaults = {
        "model_path": './models/logo_recognition/yolo/yolo_model.h5',
        "anchors_path": './models/logo_recognition/yolo/yolo_anchors.txt',
        "classes_path": './models/logo_recognition/yolo/data_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        # self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        end = timer()
        print('{} model, anchors, and classes loaded in {:.2f}sec.'.format(model_path, end-start))
        # self.yolo_model.summary(line_length=150)

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)

        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image).astype('float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
           [self.boxes, self.scores, self.classes],
           feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

#        out = self.yolo_model(image_data, training=False)
#        out_boxes, out_scores, out_classes = yolo_eval(
#            out,
#            self.anchors,
#            num_classes=len(self.class_names),
#            image_shape=[image.size[1], image.size[0]],
#            score_threshold=self.score,
#            iou_threshold=self.iou
#        )

        out_prediction = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if top >= bottom or left >= right:
                continue

            # image was expanded to model_image_size: make sure it did not pick
            # up any box outside of original image (run into this bug when
            # lowering confidence threshold to 0.01)
            if top > image.size[1] or right > image.size[0]:
                continue

            # output as xmin, ymin, xmax, ymax, class_index, confidence
            out_prediction.append([left, top, right, bottom, c, score])

        return out_prediction


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--detect_thres", type=float, default = 0.1)
    parser.add_argument("--gpu_id", type=str, default = "0")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    yolo = YOLO()
    # print(yolo.boxes, yolo.scores, yolo.classes)
    img = Image.open('../data/custom/images/train/aldi_img000028_6.jpg')
    print(img.size)
    res = yolo.detect_image(img)
    # get the real (x0, y0, x1, y1, cls, score) on the real img

    print(res)