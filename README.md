# head-circumference

## Description

This code presents a method to automatically estimate the head-circumference of a fetus from ultrasound images. It utilises a deep neural network with a [U-Net](https://en.wikipedia.org/wiki/U-Net) architecture. Importantly, there are no explicit shape constraints built-into the model, which is in contrast to the methods reviewed by [Bohlender and colleagues (2022)](https://arxiv.org/abs/2101.07721).

## Requirements

The data is from [zenodo](https://zenodo.org/record/1327317#.ZCqRwbxBzCk).

The code runs in python using [pytorch](https://pytorch.org). 
In addition to installing the necessary packages (see "import" commands in hc\_estimation\_with\_unet.py and [ellipse.py](ellipse.py)), [early-stopping](https://github.com/Bjarten/early-stopping-pytorch) code needs to be added to the folder.


## References
The data are provided by [van der Heuvel and colleagues (2018)](https://doi.org/10.1371/journal.pone.0200412) and were made available as part of a [grand challenge](http://hc18.grand-challenge.org). A version of this code was submitted as a project during the Deep-Learning Course (Data Scientist Certificate) from [alfatraining](https://www.alfatraining.com). The U-Net architecture was developed by [Ronneberger and colleagues (2015)](https://arxiv.org/abs/1505.04597) and implemented in pytorch by [Pavel Iakubovski](https://github.com/qubvel/segmentation_models.pytorch). The model implementation is inspired by Maxim Kovito [Nerve Segmentation with Ultrasound]. The fitting of the ellipse and the calculation of circumference are from code by Christian Hill [Scipython.com]  

## Output Images
Example of training data:

Example of validation data:

Loss and Accuracy:

## Links

[home page](www.magezi.com)
