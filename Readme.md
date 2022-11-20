<p align="center">
<h1>
  <img style="vertical-align:middle" width="10%" src="img/PCSkeletor_left.png">
  <b style="">PC Skeletor - Point Cloud Skeletonization</b>
  <img style="vertical-align:middle" width="10%" src="img/PCSkeletor.png">
</h1></p>

PC Skeletor is python library for extracting a 1d skeleton from point clouds using eiter the algorithm of
[Laplacian-Based Contraction](https://taiya.github.io/pubs/cao2010cloudcontr.pdf) or 
[L1-Medial Skeleton](https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf) (Not yet implemented!).

## About

Make sure that you have a Python version >=3.7 installed.

## Installation

This repository is tested on Python 3.6+ and can be installed from PyPi.
 ````bash
 pip install pc_skeletor
 ````

## Usage
````python
import pc_skeletor

# Tbd

````

## Literature and Code used for implementation

#### Laplacian based contraction

Paper: https://taiya.github.io/pubs/cao2010cloudcontr.pdf

Source Code: https://github.com/taiya/cloudcontr

````bash
@inproceedings{cao_smi10,
author = {Junjie Cao and Andrea Tagliasacchi and Matt Olson and Hao Zhang and Zhixun Su},
title = {Point Cloud Skeletons via Laplacian-Based Contraction},
booktitle = {Proc. of IEEE Conf. on Shape Modeling and Applications},
year = 2015}
````

#### L1-Medial Skeleton of Point Cloud (NOT YET IMPLEMENTED!)

Paper: https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf

Source Code: https://github.com/HongqiangWei/L1-Skeleton

````bash
@ARTICLE{Huang2013,
title = {L1-Medial Skeleton of Point Cloud},
author = H. Huang and S. Wu and D. Cohen-Or and M. Gong and H. Zhang and G. Li and B.Chen},
journal = {ACM Transactions on Graphics},
volume = {32},
issue = {4},
pages = {65:1--65:8},
year = {2013}
}
````

#### Robust Laplacian for Point Clouds

Computation of the discrete laplacian operator the code below is used.

Paper: http://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf
Source Code: https://github.com/nmwsharp/robust-laplacians-py

````bash
@article{Sharp:2020:LNT,
  author={Nicholas Sharp and Keenan Crane},
  title={{A Laplacian for Nonmanifold Triangle Meshes}},
  journal={Computer Graphics Forum (SGP)},
  volume={39},
  number={5},
  year={2020}
}
````

