# Remote Desktop

![Remote Desktop](../images/rd_desktop.png)

# What is Remote Desktop?

Remote Desktop provides an in-browser GUI Ubuntu desktop experience as well as
quick access to supporting tools. The operating system is
[**Ubuntu**](https://ubuntu.com/about) **18.04** with the
[**XFCE**](https://www.xfce.org/about) desktop environment.

## Versions

Two versions of Remote Desktop are available. _R_ includes R and RStudio.
_Geomatics_ extends _R_ with QGIS and various supporting libraries. You may
further customize your Remote Desktop workspace to suit your specific needs and
preferences.

## Customization

_pip_, _conda_, _npm_ and _yarn_ are available to install various packages.

# Accessing the Remote Desktop

To launch the Remote Desktop or any of its supporting tools, create a Notebook
Server in [Kubeflow](./Kubeflow.md) and select one of the available versions in
the image dropdown. Then, click `Connect` to access the landing page.

![Landing Page](../images/rd_landing_page.png)

_Remote Desktop_ brings you to the Desktop GUI through a noVNC session. Click on
the < on the left side of the screen to expand a panel with options such as
fullscreen and clipboard access.

![NoVNC Panel](../images/rd_novnc_panel.png)

## Legacy Interface: Jupyter Notebook

_Jupyter Notebook_ is a legacy interface for managing Jupyter Notebooks until
storage integration with the [JupyterLab images](./Jupyter.md) becomes
available.

![Jupyter Notebook](../images/rd_jupyter.png)

You can access other tools and interfaces with the _Open Tool_ button in the top
right corner, or by returning to the landing page.

![Open Tool](../images/rd_open_tools.png)

# In-browser Tools

## VS Code

_VS Code_ brings you to the Visual Studio Code IDE.

![VS Code](../images/rd_vs_code.png)

## Netdata

_Netdata_ delivers in-depth interactive resource monitoring. Charts can be
panned by dragging them. You can also zoom in and out with
`SHIFT + mouse wheel`, or zoom to a selection by holding down `SHIFT` while
dragging.

![Netdata](../images/rd_netdata.png)

## Filebrowser

_Filebrowser_ can be used to quickly explore the file system of the Remote
Desktop as well as to transfer files between the Remote Desktop and your
computer.

![Filebrowser](../images/rd_filebrowser.png)

## Access Port

Finally, _Access Port_ provides in-browser access to anything served on a
specified port inside the Remote Desktop (internally accessible at e.g.
`localhost:8059`).

![Access Port](../images/rd_access_port.png)

As with the other tools, this opens in a separate page in your browser. Your
namespace collaborators can also access this page if you send them the URL.

_Example: Accessing the Supervisor API at port 8059_

![Supervisor API](../images/rd_supervisor.png)

# Footnotes

Remote Desktop is based on
[ml-tooling/ml-workspace](https://github.com/ml-tooling/ml-workspace).
