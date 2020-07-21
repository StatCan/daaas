# Jupyter

## Friendly R and Python experience

Jupyter gives you **notebooks** to write your code and make visualizations. You
can quickly iterate, visualize, and sahre your analyses. Because it's running on
a server (that you set up in the last section) you can do really big analyses on
centralized hardware! Adding as much horsepower as you need! And because it's on
the cloud, you can share it with your colleagues too.

### Explore your data

Jupyter comes with a number of features (and we can add more)

- Integrated visuals within your notebook
- Data volume for storing your data
- You can share your workspace with colleagues.

![Interactive Widgets](../images/jupyter_visual.png)

### IDE in the browser

Create for exploring, and also great for writing code

- Linting and a debugger
- Git integration
- Built in Terminal
- Light/Dark theme (change settings at the top)

![IDE features](../images/jupyter_ide.png)

**More information on Jupyter [here](https://jupyter.org)**

## Get started with the examples

When you started your server, it got loaded with a bunch of example notebooks.
Great notebooks to start with are `R/01-R-Notebook-Demo.ipynb`, or the notebooks
in `scikitlearn`. `pytorch` and `tensorflow` are great if you are familiar with
machine learning. The `mapreduce-pipeline` and `ai-pipeline` are more advanced.

<!-- prettier-ignore -->
??? danger "Some notebooks only work in certain server versions"
    For instance, `gdal` is only in the geomatics image. So if you use another
    image then a notebook using gdal might not work.

## Adding software

You do not have `sudo` in Jupyter, but you can use

```sh
conda install --use-local your_package_name
```

or

```sh
pip install --user your_package_name
```

**Don't forget to restart your jupyter kernel afterwards, to make new packages
available.**

<!-- prettier-ignore -->
??? tip "Make sure to restart the Jupyter kernel after installing new software"
    If you install software in a terminal, but your jupyter kernel was already
    running, then it will not be updated.

<!-- prettier-ignore -->
??? tip "Is there something that you can't install?"
    If you need something installed, reach us or
    [open a GitHub issue](https://github.com/StatCan/kubeflow-containers). We
    can add it to the default software.

# Getting Data in and out of Jupyter

You can upload and download data to/from Jupyterhub directly in the menu. There
is an upload button at the top, and you can right-click most files or folders to
download them.

## Shareable "Bucket" storage

**The other option** is high-volume storage with
[Object Storage](https://en.wikipedia.org/wiki/Object_storage). Because storage
is important for experiments, publishing, and exploring datasets, it has its own
section.

**Refer to the [Storage Section](/Storage)**
