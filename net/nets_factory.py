from tensorflow.keras.applications import Xception, VGG16, VGG19
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetMobile


#for Contains a factory for building various models.
networks_map = {'vgg16': VGG16,
                'vgg19': VGG19,
                'inception_v3': InceptionV3,
                'inception_resnet_v2': InceptionResNetV2,
                'resnet50': ResNet50,
                'resnet101': ResNet101,
                'resnet152': ResNet152,
                'resnet50_v2': ResNet50V2,
                'resnet101_v2': ResNet101V2,
                'resnet152_v2': ResNet152V2,
                'mobilenet_v1': MobileNet,
                'densenet121' : DenseNet121,
                'densenet169' : DenseNet169,
                'densenet201' : DenseNet201,
                'NASNet': NASNetMobile,
                'xception' : Xception,
               }


def get_network(network):
    model = networks_map[network]
    return model
