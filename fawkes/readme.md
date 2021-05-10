This is an introduction to the files in the fakes folder。All the resources in the fawkes directory are referenced to [website](https://github.com/Shawn-Shan/fawkes).

# Usage

$ fawkes

## Options:
- **-d**,**directory**: the direcotory that contains images to run protection, and the default is 'imgs/'.
- **-g**,**--gpu**: the GUP id when using GPU for optimization, and the default is '0'.
- **-m**,**--mode** : the tradeoff between privacy and perturbation size. Select from **min**,**low**,**mid**,**high**. The higher the mode is, the more perturbation will add to the image and provide stronger protection.
- **feature-extractor**: name of the feature extractor used for optimization, and the default is "arcface_extractor_0".

- **--batch-size**: number of images to run optimization together. Change to >1 only if you have extremely powerful compute power.
- **--format**: format of the output image(png or jpg).

## Example
fawkes -d ./imgs --model low -g 0 --batch-size 1 --format png

or python protection.py -d ./imgs --model low --batch-size 1 --format png

