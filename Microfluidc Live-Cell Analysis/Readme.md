# Automated Live-Cell Image Analysis

With the skripts in this repository the result plot shown in Fig. 4A can be repoduced.

## Setup your environment

1. Make sure you have anaconda installed. If you do not have it get from [here](https://www.anaconda.com/products/individual).
2. Make sure you have [git-lfs](https://git-lfs.github.com/) installed.
3. Install all the software dependencies by setting up an environment in conda. You can do this by executing the following lines in the command line: 
    ```bash
    conda env create -f conda.yml
    conda activate conos-analysis
    ```
    The installation might take some minutes.

## Analysis
To perform the analysis execute
```python
python fluorescence_analysis.py input.tif
```

You will find the plots and videos in the `output` folder.

**Note**: Due to the long image sequence (140 frames) we recommend having at least 32 GB of RAM on your computer.
