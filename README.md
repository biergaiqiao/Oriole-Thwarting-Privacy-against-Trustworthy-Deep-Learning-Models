# Oriole-Thwarting-Privacy-against-Trustworthy-Deep-Learning-Models
This project is prepared for the paper "Oriole: Thwarting Privacy against Trustworthy Deep Learning Models" published in ACISP.

## Copyright
This code is intended only for presonal privacy or academic reserch. We prohibit commercialization activities without permisssion. The copyright belongs to the master student [Liuqiao Chen](https://dblp.org/pid/286/1713) and Prof. [Qian](https://dblp.org/pid/61/6767) of East China Normal University.

## Fawkes
This folder contains relevant resources in this paper([Fawkes: Protecting Privacy against Unauthorized Deep Learning Models](https://www.usenix.org/conference/usenixsecurity20/presentation/shan)). More details in the relevant folder [fawkes](https://github.com/biergaiqiao/Oriole-Thwarting-Privacy-against-Trustworthy-Deep-Learning-Models/tree/main/fawkes).

### Quick Intallation
`pip install fawkes'


## Oriole
This folder contains relevant resources in this paper([Oriole: Thwarting Privacy against Trustworth Deep Learning Models](https://arxiv.org/abs/2102.11502)). More details in the relevant folder [oriole](https://github.com/biergaiqiao/Oriole-Thwarting-Privacy-against-Trustworthy-Deep-Learning-Models/tree/main/oriole).

### Tips
- Your'd better not change the value of the batch-size unless you have very powerful GPU computing resources.
- Run on GPU. neither the current Fawkes packge or the Oriole package support GPU. To use GPU, you need to clone this repo, install the required packges in **setup.py**, and replace the **tensorflow** with **tensorflow-gpu** Then you can run Fawkes or Oriole like this:
 **python fawkes/protection.py [args]** (Fawkes) 
 or **python oriole/F_protection.py [args]**(Oriole).
 
 ![exmaple](https://github.com/biergaiqiao/Oriole-Thwarting-Privacy-against-Trustworthy-Deep-Learning-Models/tree/main/material/example.png).
