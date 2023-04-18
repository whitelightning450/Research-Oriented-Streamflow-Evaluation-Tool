# Script to access NOAA NWM data from AWS S3
# Author: Karnesh Jain

import xarray as xr
import s3fs
from datetime import datetime


def get_nwm_data(feature_id, start_date, end_date):
    """
    Get NOAA NWM data from AWS
    It is filtered to retrieve data for a particular time range corresponding to a feature ID
    Arguments:
    ----------
    feature_id (int): Feature ID for which NWM data needs to be returned
    start_date (str): Start date in "YYYY-MM-DD" format
    end_date (str): End date in "YYYY-MM-DD" format
    Returns
    -------
    (pandas.dataframe): Pandas dataframe with NWM data for user queried time range and feature ID
    """

    # check start and end date format
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Start and end date should have YYYY-MM-DD format")

    url = "s3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr"
    fs = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(url, s3=fs)
    ds_nwm_chrtout = xr.open_zarr(store, consolidated=True)

    ds_nwm_filtered = ds_nwm_chrtout.sel(feature_id=feature_id, time=slice(start_date, end_date))

    df_nwm_chrtout = ds_nwm_filtered.to_dataframe()

    return df_nwm_chrtout