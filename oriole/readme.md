This is an introduction to the files in the oriole folder。

# Usage
$ fawkes
## Options:
- **--power**: the numbers of multi-cloaks for each uncloaked image, we use parameter *m* to represent this in the [paper](https://arxiv.org/abs/2102.11502), and default is 1. 
- **--DEEP**: stand for a tag, if the value is 1, then we can only process all the pictures in the first-level directory, otherwise, we can process all the prictures in the second-level directory, and the default value is 1.
- **-d**,**directory**: the direcotory that contains images to run protection, and the default is 'imgs/'.
- **-g**,**--gpu**: the GUP id when using GPU for optimization, and the default is '0'.
- **-m**,**--mode** : the tradeoff between privacy and perturbation size. Select from **min**,**low**,**mid**,**high** or **custom**. The higher the mode is, the more perturbation will add to the image and provide stronger protection.
- **feature-extractor**: name of the feature extractor used for optimization, and the default is "arcface_extractor_0".
- **--th**: only relevant with mode "custom", DSSIM threshold for perturbation, and default is 0.01.
- **--max-step**: only relevant with mode "custom", number of steps for optimization, and default is 1000.
- **--sd**: only relevant with mode "custom", penalty number, read more in the [paper](https://www.usenix.org/conference/usenixsecurity20/presentation/shan), and the default is 1e6.
- **--lr**: only relevant with mode "custom", learning rate,default is 2.
- **--batch-size**: number of images to run optimization together. Change to >1 only if you have extremely powerful compute power, and the default is 1.
- **--separate_target**: whether select separate targets for each faces in the directory, and default is 'True'.
- **--no-align**: whether to detect and crop faces, default is 'True'.
- **--debug**: run on debug and copy/paste the stdout when reproting and issue on github, and default is 'True'.
- **--format**: format of the output image(png or jpg), and default is 'png'.

## Example
`fawkes --DEEP 1 --power 2 -d ./imgs --model low -g 0 --batch-size 1 --format png`

or `python protection.py --DEEP 1 --power 2 -d ./imgs --model low --batch-size 1 --format png`

## Tips
After you download this resources in the directory **'oriole'**, first you need to copy the file **'utils.py'** to the path of **'fawkes'** in your corresponding virtual environment to overwrite the file **'utils.py'**, and then these will work.
