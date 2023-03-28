# PC-Skeletor - Point Cloud Skeletonization
<br/>

<img src="_static/lbc.gif" width="46%"/> <img src="_static/s_lbc.gif"  width="46%"/>

<br/>

## 📖 Abstract
Standard Laplacian-based contraction (LBC) is prone to mal-contraction in cases where
there is a significant disparity in diameter between trunk and branches. In such cases fine structures experience 
an over-contraction and leading to a distortion of their topological characteristics. In addition, LBC shows a 
topologically incorrect tree skeleton for trunk structures that have holes in the point cloud.In order to address 
these topological artifacts, we introduce semantic Laplacian-based contraction (S-LBC). It integrates semantic 
information of the point cloud into the contraction algorithm

```text
 @inproceedings{schoenberger2016sfm,
        author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
        title={Structure-from-Motion Revisited},
        booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2016},
    }
```


## ⚡️ Quick Start

### Installation

First install [Python](https://www.python.org/downloads/) Version 3.7 or higher. The python package can be installed
via [PyPi](https://pypi.org/project/pc-skeletor/) using pip.

 ````bash
pip install pc-skeletor
 ````

### Installation from Source

 ````bash
git clone https://github.com/meyerls/pc_skeletor.git
cd pc_skeletor
pip install --upgrade pip setuptools
pip install -r requirements.txt
pip install -e .
 ````