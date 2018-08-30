# analyzing cyclists counts in Auckland using a Generalized Additive Model (`fbpropet`) 

## installation instructions 

I assume you all have the [Anaconda Python Distribution](https://www.anaconda.com/download/) installed 

You'll need the following packages installed on top of the defaults ones: 

+ [seaborn](https://seaborn.pydata.org/) for some visualisation, can be installed either from the Anaconda navigator, or the command line: 

  ```
  conda install seaborn
  ```

+ [folium]() for creating the interactive map of the cycling counters location (optional), we need the latest development version, installable via pip on the command line: 

  ```
  pip install git+https://github.com/python-visualization/folium
  ```

+ and of couse [fbprophet](https://facebook.github.io/prophet/) ... it is conda-installable from the conda-forge channel:

  ```
  conda install -c conda-forge fbprophet
  ```