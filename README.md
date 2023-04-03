# head-circumference

## Description

This code presents a method to automatically estimate the head-circumference of a fetus from ultrasound images. It utilises a deep neural network with a [U-Net](https://en.wikipedia.org/wiki/U-Net) architecture. Importantly, there are no explicit shape constraints built-into the model, which is in contrast to the methods reviewed by [Bohlender and colleagues (2022)](https://arxiv.org/abs/2101.07721).

## Requirements

The data is from [zenodo](https://zenodo.org/record/1327317#.ZCqRwbxBzCk).

The code runs in python using [pytorch](https://pytorch.org). 
In addition to installing the necessary packages (see "import" commands in hc\_estimation\_with\_unet.py and ellipse.py),[early-stopping](https://github.com/Bjarten/early-stopping-pytorch) code needs to be added to the folder.


## Acknowledgements
The data are provided by van der Heuvel et al. (2018) 

## Output Images
Example of training data:

Example of validation data:

Loss and Accuracy:

## Links

[home page](www.magezi.com)
