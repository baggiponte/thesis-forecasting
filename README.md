# A Comparison of Forecasting Methods for Bike Sharing Services

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm.fming.dev)

[![CodeFactor](https://www.codefactor.io/repository/github/baggiponte/thesis-forecasting/badge?s=15440e769166c9c9571f2d7b8d3c7fcdcf5c9ecb)](https://www.codefactor.io/repository/github/baggiponte/thesis-forecasting)

This is the repository for my graduate thesis on time series forecasting, comparing the accuracy of canonical statistical models (such as ARIMA) with more recent approaches: Auto-ARIMA, Prophet by Meta (Facebook) and machine learning models (with ensemble techniques).

## Structure of the repo

```bash.
.
├── README.md
├── body
├── data
│   ├── milan
│   └── queries
├── environment.yml
├── notebooks
│   ├── 01-introduction.ipynb
│   ├── 02-bikesharing_and_bikemi.ipynb
│   ├── 03-data_ingestion_and_spatial_operations.ipynb
│   ├── 04-data_transform_and_stationarity.ipynb
│   ├── 04-time_series_analysis.ipynb
│   ├── 05-ARIMA.ipynb
│   ├── 06-Prophet.ipynb
│   ├── 07-extensions.ipynb
│   ├── README.md
│   └── custom_functions
└── scripts
```

* [`README.md`](https://github.com/baggiponte/thesis-forecasting/tree/main/README.md) is this file.
* [`body`](https://github.com/baggiponte/thesis-forecasting/tree/main/body) contains the MyST (Markedly Structured Text) notebooks used to convert the Jupyter Notebooks into the PDF book.
    * The compiled `.pdf` is [here](https://github.com/baggiponte/thesis-forecasting/blob/main/body/_build/latex/thesis-bike_sharing_forecasting.pdf)
* [`data`](https://github.com/baggiponte/thesis-forecasting/tree/main/data) contains the data that could be disclosed. The private data is stored on a private database.
  * [`milan`](https://github.com/baggiponte/thesis-forecasting/tree/main/data/milan) contains the open data about Milan that I collected and manipulated
  * [`queries`](https://github.com/baggiponte/thesis-forecasting/tree/main/data/queries) contains the SQL queries to retrieve the data from the server and create materialised views.
* [`environment.yml`](https://github.com/baggiponte/thesis-forecasting/tree/main/environment.yml) is the specification files to reproduce this project.
* [`notebooks`](https://github.com/baggiponte/thesis-forecasting/tree/main/environment.yml) are the notebooks for the analysis.
* [`scripts`](https://github.com/baggiponte/thesis-forecasting/tree/main/scripts) are the bash scripts written for this project. Since these are symlinked in my script folder, you won't find them here. However, I put that scripts folder under version control, and has its own repo [here](https://github.com/baggiponte/scripts).
