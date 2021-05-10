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
One face landmark detector that has proven to work well in this setting is the [Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). 

#### Set the python paths
Set the environment variable `PYTHONPATH` to point to the `src` directory of the cloned repo. This is typically done something like this

`export PYTHONPATH=[...]/facenet/src`
where [...] should be replaced with the direcotry where cloned facenet  repo resides.
