# Loading data into Power BI

We do not offer a Power BI server, but you can pull your data into Power BI from
our Storage system, and use the data as a `pandas` data frame.

![Power BI Dashboard](../images/powerbi_dashboard.png)

## What you'll need

1. A computer with Power BI, and Python 3.6
2. Your MinIO `ACCESS_KEY` and `SECRET_KEY` on hand. (See
   [Storage](/daaas/en/Storage))

## Get connected

### Set up Power BI

Open up your Power BI system, and open up this
[Power BI quick start](https://raw.githubusercontent.com/StatCan/jupyter-notebooks/master/querySQL/power_bi_quickstart.py)
in your favourite text editor.

You'll have to make sure that `pandas`, `boto3`, and `numpy` are installed, and
that you're using the right Conda virtual environment (if applicable).

![Install the dependencies](../images/powerbi_cmd_prompt.png)

You'll then need to make sure that Power BI is using the correct Python
environment. This is modified from the options menu, and the exact path is
specified in the quick start guide.

### Edit your python script

Then, edit your Python script to use your MinIO `ACCESS_KEY` and `SECRET_KEY`,
and then click "Get Data" and copy it in as a Python Script.

![Run your Python Script](../images/powerbi_python.png)
