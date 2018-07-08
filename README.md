# Implementation of a Framework to obtain depth Images from RGB Images using Deep Learning

This work implements the architecture laid out in the paper by Li, Jun, Reinhard Klein, and Angela Yao. "Learning fine-scaled depth maps from single RGB images." arXiv preprint (2016). 
This implementation has equal contributions<sup>1</sup> from [Neha Das](https://github.com/neha191091), [Sumit Dugar](https://github.com/dugarsumit), [Saadhana Venkatraman](https://gitlab.lrz.de/ga83pof) and [San Yu Huang](https://gitlab.lrz.de/ga59hoc).

The experiments from this work are summarized in the following poster (PDF version [here](https://github.com/neha191091/rgb-depth/blob/master/documents/Poster_v4.pdf)
![Poster](https://github.com/neha191091/rgb-depth/blob/master/documents/Poster_v4.jpg)

## Prerequisites
You will require Pytorch and Jupyter Notebook for running this project.

## Datasets
The training and evaluation of the architectures in this project were performed using RMRC Indoor Depth Chalenge Dataset. You can dowload it [here](http://cs.nyu.edu/silberman/rmrc2014/indoor.php)

## Running the experiments
The experiments in this project can be run by executing **launcher.py**.

<sup>1</sup>This implementation was imported from [https://gitlab.lrz.de/](https://gitlab.lrz.de/) and therefore has an improper commit history in github. The original project can be accessed [here](https://gitlab.lrz.de/dugarsumit/dlcv_proj)

