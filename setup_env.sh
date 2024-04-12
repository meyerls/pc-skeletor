conda create --name s-lbc python=3.8 -y
conda activate s-lbc
conda install conda-forge::gfortran
pip install --upgrade pip setuptools
pip install -r requirements.txt
