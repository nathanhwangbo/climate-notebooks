import re
import regionmask
import numpy as np
from pathlib import Path
import requests
import fsspec
import polars as pl
from datetime import date
import xarray as xr

# where to put the example data
data_path = Path("data")


def add_member_dim(x):
    ensemble_member = int(re.search(r"\.(\d+)\.", x.encoding["source"]).group(1))
    # Add a new dimension and coordinate for the ensemble member
    x = x.expand_dims("member").assign_coords({"member": [ensemble_member], "lat": x.lat.round(3)})
    x = x["AODVIS"]
    return x


def add_euus_landmask(ds, us_only=False, eu_only=False):
    """
    add landmask for europe and united states.
    uses pretty broad boxes, which includes bits of north america outside of the US.

    :param: ds is an xarray object with "lon" and "lat" coordinates, where lon is (-180, 180)
    :param: us_only, europe_only. flag for including only one or the other. default: include both.
    """
    # create a landmask
    land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    landmask = land.mask(ds)  # ocean is nan, land is 0
    is_land = landmask == 0

    # also get rid of greenland
    greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110[["Greenland"]]
    gl_mask = greenland.mask(ds)
    is_not_greenland = gl_mask.isnull()

    # is us (ish)
    is_usa_lat = np.logical_and(ds["lat"] >= 25, ds["lat"] <= 50)
    is_usa_lon = np.logical_and(ds["lon"] >= -125, ds["lon"] <= -65)
    is_usa = np.logical_and(is_usa_lat, is_usa_lon)

    # is europe (ish)
    is_eu_lat = np.logical_and(ds["lat"] >= 35, ds["lat"] <= 60)
    is_eu_lon = np.logical_and(ds["lon"] >= -15, ds["lon"] <= 50)
    is_eu = np.logical_and(is_eu_lat, is_eu_lon)

    if us_only:
        is_in_region = is_usa
    elif eu_only:
        is_in_region = is_eu
    else:
        # is in us or europe
        is_in_region = np.logical_or(is_usa, is_eu)

    # apply landmask
    ds = ds.where(is_land & is_not_greenland & is_in_region)

    return ds


def download_aod(path):
    """
    about 1.5gb of example data, (first ten members of aerosol optical depth from LENS)
    """
    fs = fsspec.filesystem("http")
    base_url = "http://bickford.asuscomm.com:3923/CESM1/LENS_monthly_AODVIS/"
    resp = requests.get(base_url, params={"q": "ext:nc", "ls": ""})
    data = resp.json()

    lens_urls = [f"{base_url}/{f['href']}" for f in data["files"]]
    # grab only the first ten members
    one_to_ten = {f".{i:03d}." for i in range(1, 11)}
    first_ten_members = [url for url in lens_urls if any(m in url for m in one_to_ten)]

    # leave out the last 20 years
    url_list = [url for url in first_ten_members if not url.endswith("208101-210012.nc")]

    # WARNING: downloading files to data_path!
    for url in url_list:
        target_path = path / Path(url).name
        if not target_path.exists():
            print(f"downloading {target_path}")
            fs.get(url, target_path)


def generate_ensemble_xr(
    n_members=10,
    start_year=2000,
    end_year=2010,
    interval="1mo",
    n_lat=20,
    n_lon=40,
    noise_sd=2,
    rng=np.random.default_rng(seed=123),
):
    """
    crude function to generate some synthetic data with a yearly wiggle and some pole-to-pole activity

    Parameters
    ----------
    n_members : int, optional
        number of "ensemble members", by default 10
    start_year : int, optional
        year the data will start, by default 2000
    end_year : int, optional
        year the data will end, by default 2010
    interval : str, optional
        polars options for intervals, by default "1mo"
    n_lat : int, optional
        number of lat bands between -60 and 60, by default 20
    n_lon : int, optional
        number of lon bands between -180 and 180, by default 40
    noise_sd : float, optional
        sd of the noise for each member, by default 2
    rng : numpy generator, optional
        if you want consistent rng, by default np.random.default_rng(seed=123)

    Returns
    -------
    two xarray datasets: one with the synthetic ensemble, and another with the "signal"
        _description_
    """
    times = pl.date_range(date(start_year, 1, 1), date(end_year, 12, 1), interval=interval, eager=True)
    lat = np.linspace(-60, 60, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    members = range(n_members)
    n_time = len(times)

    # add some wiggle
    if interval == "1mo":
        time_idx = np.arange(n_time)
        year_wiggle = np.cos(2 * np.pi * time_idx / 12)

        spatial_wiggle = np.cos(np.deg2rad(lat))[:, None] * np.sin(np.deg2rad(lon))[None, :]
        signal = year_wiggle[:, None, None] * spatial_wiggle
        signal_ds = xr.DataArray(signal, coords=[times, lat, lon], dims=["time", "lat", "lon"])
    else:
        print("implement other intervals later!")

    obs_list = []
    for m in members:
        # add a lot of noise
        noise = rng.normal(0, noise_sd, size=(n_time, n_lat, n_lon))
        noise_ds = xr.DataArray(noise, coords=[times, lat, lon], dims=["time", "lat", "lon"])
        obs = signal_ds + noise_ds
        obs_list.append(obs)

    da_ensemble = xr.concat(obs_list, dim="member")
    da_ensemble.name = "x"

    return da_ensemble, signal_ds


def generate_ensemble_ts(start_year=1900, end_year=2000, noise_sd=0.5, n_members=10, interval="1mo"):
    # create synthetic ts with seasonal cycle
    start_date = date(start_year, 1, 1)
    end_date = date(end_year, 12, 1)

    signal_df = (
        pl.DataFrame({"date": pl.date_range(start_date, end_date, interval=interval, eager=True)})
        .with_columns(
            month=pl.col("date").dt.month(),
            trend=pl.int_range(0, pl.len()) * 0.01,  # small trend
        )
        .with_columns(
            # -cos gives us a "winter low"
            # month - 1 so Jan is at the bottom
            seasonality=-((pl.col("month") - 1) / 12 * 2 * np.pi).cos() * 5
        )
        .with_columns(
            signal=pl.col("trend") + pl.col("seasonality")
            # signal=pl.col("seasonality")
        )
    )

    # generate members by adding noise
    ensemble_list = []
    for i in range(n_members):
        member_noise = np.random.normal(0, noise_sd, signal_df.height)
        member_df = signal_df.select(
            pl.col("date"),
            pl.lit(i).alias("member"),
            (pl.col("signal") + member_noise).alias("x"),
        )
        ensemble_list.append(member_df)

    ens_df = pl.concat(ensemble_list)
    return ens_df, signal_df
