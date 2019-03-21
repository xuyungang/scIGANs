# scIGANs (v0.1.1)
Generative adversarial networks for single-cell RNA-seq imputation
## Table of Contents
- [Introduction](#introduction)
- [Installation](#install)
  - [Operating system](#os)
  - [Install Dependences](#depend)
  - [Install scIGANs](#build)
- [Run scIGANs](#run)
  - [Commends and Options](#cmd)
  
## <a name="introduction"></a>Introduction
scIGANs is ...
## <a name="install"></a>Installation
### <a name="os"></a>Operating system
scIGANs currently can only be built and run on Linux/Unix systems.
### <a name="depend"></a>Install dependences
scIGANs is implemented in `python` (>2.7) and `R`(>3.5). Please install `Python` (>2.7) and `R`(>3.5) before run scIGANs.
- **R packages:**  scIGANs will automatically install dependent packages. To make sure you have the permission to install R packages to the lib.
- **python modules:** `pytorch`, `numpy`, `pandas`, `torchvision`, and `joblib` are required for scIGANS.
### <a name="build"></a>Install scIGANs
- **download** `git clone https://github.com/xuyungang/scIGANs0.1.1.git`
- `cd scIGANs0.1.1`
- **install** `bash scIGANs.install [-p dir/to/install]`
  - use `-p` option to direct the installation to a sepecific directory; default, current working directory.
- **check installation** `scIGANs -h`
  - For successful installation, you will see the help message.
  - Occasionally, you may need to restart you terminal to run **`scIGANs`**.

 ## <a name="run"></a>Run scIGNAs
 ### <a name="cmd"></a>Commands and options
```
Usage:   scIGAN in.matrix.txt [options]

Options:
         -h --help       Show this usage message. 
    input:
         in.matrix.txt   A tab-delimited text file, containing the expression counts matrix with genes in 
                         rows and cells in columns. The  first row is header and first column is gene IDs
                         or names. <required>        
         -l --label_file [STR]  A text file contain the labels (cell types, subpopulations), 
                                each per line with the same order in in.matrix.txt. <optional> 
                                Default: scIGANs will learn the subpopulations using Spectral clustering.
    training:
         -n --n_epochs   [STR]   The number of epochs to train the GANs. <optional> Default: 200
         -p --process    [STR]   Number of threads to run scIGANs. <optional> Default: 20
    imputing:
         -s --sim_szie   [INT]   Number of generated datasets for imputing. <optional> Default: 200
         -k --knn_n      [INT]   Number of nearest neighbours for imputing. <optional> Default: 10
    Output:
         -o --outdir     [STR]   output directory. <optional> Default: current working directory
```
