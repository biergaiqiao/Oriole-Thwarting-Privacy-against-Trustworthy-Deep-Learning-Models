# Face Recognition using Tensorflow
This is a repo about how we train our model, which refers to resource [Face Recognition using tensorflow](https://github.com/davidsandberg/facenet#face-recognition-using-tensorflow-).

## Pretrained models
| Model name | LFW accuracy | Training dataset | Architecture |
|--------|--------|--------|--------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905 | CASIA-WebFace | [Inception ResNet V1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)| 0.9965 | VGGFace2 | [Inception ResNet V1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## Inspriation
The code is heavily inspired by the [Facenet](https://github.com/davidsandberg/facenet) implementation.

## Pre-processing
### Face alignment using MTCNN

