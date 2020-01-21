# scIGANs (v0.1.1)
Generative adversarial networks for single-cell RNA-seq imputation
## Table of Contents
- [Introduction](#introduction)
- [Installation](#install)
  - [Operating system](#os)
  - [Install scIGANs](#build)
  - [Install Dependences](#depend)
- [Use scIGANs](#run)
  - [Commends and Options](#cmd)
  - [Input file formate](#input)
  - [Output file](#output)
  - [Run with test data](#test)
- [Contact](#contac)
- [Comments and Bugs](#issue)
- [Citation](#cite)
## <a name="introduction"></a>Introduction
scIGANs is a computational tool for single-cell RNA-seq imputation and denoise using Generative Adversarial Networks (GANs). Build on pythorch, scIGNAs enables GPU acceleration inaddition to CPU computing as long as the GPUs are available.
## <a name="install"></a>Installation
### <a name="os"></a>Operating system
scIGANs currently can only be built and run on Linux/Unix-based systems.
### <a name="build"></a>Install scIGANs
- **Download** `git clone https://github.com/xuyungang/scIGANs.git`
- `cd scIGANs`
- **Install** `bash scIGANs.installer [-p dir/to/install]`
  - use `-p` option to direct the installation to a sepecific directory; default, current working directory.
  - You may need to start a new terminal or `source ~/.bashrc` to enable `scGANs` executable from your `$PATH`.
- **Check installation** `scIGANs -h`
  - For successful installation, you will see the help message.
  - Occasionally, you may need to restart you terminal to run **`scIGANs`**.
  - If you installed scIGANs before, please use `which scIGANs` to check the currently active version.
### <a name="depend"></a>Install dependences
scIGANs is implemented in `python` (>2.7) and `R`(>3.5). Please install `Python` (>2.7), `R`(>3.5) and all dependencies before run scIGANs. Users can either use pre-configured conda environment (recommended) or build your own environment manually.
  #### Use pre-configured conda environment (recommended)
  1. Install conda (>4.5)
  - If you already got conda installed, go to step `2` directly.
  - Downlowd bash installer (or find different version at https://docs.conda.io/en/latest/miniconda.html)
    - `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
  - Install conda (Linux for example)
    - `bash miniconda/Miniconda3-latest-Linux-x86_64.sh`
    - follow the prompted instruction to complete the installation.
  2. Create conda envirenment for scIGANs
  - `cd scIGANs`
    
    **Note:** if you have scIGNAs conda environment configured before, please remove it before proceeding.
    
    `conda env remove -n scIGNAs`
  - For Linux: `conda env create -f scIGANs.conda.env.Linux.yml`
  - For Mac OS: `conda env create -f scIGANs.conda.env.Mac.yml`

  3. Activate conda envirenment for scIGANs
  - Activate conda environment: `conda activate scIGNAs`
  - [Test scIGNAs](#test)
  ####  Manually installation (skip this step if you finished conda configuration as above)
  - **R packages:**  scIGANs requires two R packages: `SamSPECTRAL` and `Rtsne`. Make sure you have the permission to install R packages to the default lib.
  - **python modules:** `pytorch`, `numpy`(will be installed with pytorch when using `conda install`), `pandas`, `torchvision`, and `joblib` are required for scIGANS.
## <a name="run"></a>Use scIGNAs
### <a name="cmd"></a>Commands and options

- **Usage:** `scIGAN in.matrix.txt [options]`

- **Options:**

    - `-h --help`      Show this usage message and exit scIGANs. 
    - *Input:*
        - `in.matrix.txt` \[STR\]  A tab-delimited text file (including appropriate PATH), containing the expression counts with genes in 
                         rows and cells in columns. The  first row is header and first column is gene IDs
                         or names. \<**required**\> 
        - `-l|--label_file` \[STR\]  A text file (including appropriate PATH) contains the labels (cell types, subpopulations, etc.), 
                                one label per line per cell  with the same order in in.matrix.txt. \<**optional**\> 
                                Default: scIGANs will learn the subpopulations using Spectral clustering.
    - *Training:*
        - `-e|--epochs`   \[INT\]   The number of epochs to train the GANs. \<optional\> Default: 200
        - `-d|--latent_dim` \[INT\]   Dimension of the latent space of generator. \<**optional**\> Default: 100
        - `-b|--batch_size` \[INT\]   The number of samples per batch to load.More samples per batch requires more memory.\<**optional**\> Default: 8. NOTE: no more than cell number.
        - `-j|--job_name` \[STR\]   A string will be used as prefix to name all output files.\<**optional**> Default: *prefix=\<in.matrix.txt>-<label_file_name>*
        - `-t|--threthold`  \[FLOAT\] Convergence threthold. \<**optional**> Default: 0.01.
    - *Imputing:*
        - `--impute`         Set this option to skip training and directly impute using pretrained model for the same input settings.
        - `-s|--sim_szie`   \[INT\]   Number of generated datasets for imputing. \<**optional**> Default: 200
        - `-k|--knn_n`      \[INT\]   Number of nearest neighbours for imputing. \<**optional**> Default: 10
    - *Output:*
         - `-o|--outdir`    \[STR\]   Output directory. \<**optional**> Default: current working directory
### <a name="input"></a>Input file format
scIGANs takes tab-delimited text file(s) as input. The expression count matrix file is required and needs to be imputed. The cell labels file is optional and contains the cell labels, each per row with the same cell order as in expression matrix file. The following shows the toy example formats.
![input format](test_data/scIGANs_input.png)
### <a name="output"></a>Output files
scIGANs will generate following files:
- **`Model files:`** one file for Discriminator and one file for Generator are located in thhe folder `GANs_models/`. Files are named in the format of `\<in.matrix.txt>-<label_file_name>-\<latent_dim>-\<n_epochs>-\<cluster_number>-g.pt` or `\<in.matrix.txt>-<label_file_name>-\<latent_dim>-\<n_epochs>-\<cluster_number>-d.pt`.
- **`Imputed matrix:`** expression matrix, with the same format as input and only has some zero-counts replaced with expression values imputed by scIGANs. File is named in the format of `scIGANs_\<timestamp>_\<in_matrix_filename>`.
- **`Log file:`** a log file of the running, named as `scIGANs_\<timestamp>_\<in_matrix_filename>.log`.

### <a name="test"></a>Run with test_data
- Without label file: `scIGANs scIGNAs/install/path/test_data/ercc.txt -n 1 [options]`
- With label file: `scIGANs scIGNAs/install/path/test_data/ercc.txt -l ercc.label.txt -n 1 [options]`
## <a name="contact"></a>Contact
Yungang Xu yungang.xu@uth.tmc.edu

Zhigang Zhang zzg@hbue.edu.cn

Xiaobo Zhou xiaobo.zhou@uth.tmc.edu
### <a name="issue"></a>Comment and bug report
[github issues](https://github.com/xuyungang/scIGANs0.1.1/issues)
### <a name="cite"></a>Citation
Yungang Xu, Zhigang Zhang, Lei You, Jiajia Liu, Zhiwei Fan, Xiaobo Zhou. (2019) **scIGANs: Single-cell RNA-seq Imputation using Generative Adversarial Networks** In preparation (Coming soon on BioRxiv). 
