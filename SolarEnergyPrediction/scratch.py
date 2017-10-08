"""
Interpolate one variable for all location at once
Interpolate different variabls simultanesouly on multiple threads
"""

from netCDF4 import Dataset
import numpy as np
import os
from scipy.interpolate import interp2d
import pandas as pd
from threading import Thread
import time

class OneVarInterp():

    def __init__(self, data_loc="/homes/li108/Dataset/SolarEnergy", var_idx=0):
        var_name = os.listdir(data_loc)[var_idx]
        var_nc = Dataset(os.path.join(data_loc, var_name))
        self.time = var_nc.variables["time"]
        self.lat = var_nc.variables["lat"]
        self.lon = var_nc.variables["lon"]
        self.fhour = var_nc.variables["fhour"]

        self.feature = var_nc.variables.values()[-1]
        self.feature = np.array(self.feature)
        self.feature = self.feature.mean(axis=1) # average out 11 models

        
        # Solar Power Station Info
        self.station_info = pd.read_csv("station_info.csv")
        self.station_names = self.station_info['stid']

    def get_station_position(self, station_name):
        b = self.station_info
        lat = np.asscalar(b.loc[b["stid"]==station_name]["nlat"])
        lon = np.asscalar(b.loc[b["stid"]==station_name]["elon"])
        return lon, lat # x_coord, y_coord

    def _interp_one_station(self, x_coord, y_coord, interpolators):
        """
        Arguments:
            x_coord, y_coord : longitude, latitude of the station
            interpolators : a list of interpolators, each element
                in the list correspond to one fhour
        Return:
            Interpolated 5 measurements
            A numpy array of shape (1, 5)
        """
        one_station = []
        for interp in interpolators:
            one_station.append(interp(x_coord, y_coord))
        
        return np.array(one_station).reshape(1, 5)

    def sing_day_feature_one_loc(self, time, lat, lon):
        lat_idx = np.where(self.lat.__array__() == lat)
        lon_idx = np.where(self.lon.__array__() == lon)

        return self.feature[time,:,lat_idx, lon_idx].flatten()
        

    def data_on_meshgrid(self, time):
        meshgrid_data = []
        for lat in self.lat.__array__():
            along_lat = []
            for lon in self.lon.__array__():
                along_lat.append(self.sing_day_feature_one_loc(
                                 time = time, lat=lat, lon=lon))
            meshgrid_data.append(along_lat)
        return np.array(meshgrid_data) 
        

    def interp_once(self, time):
        """
        Return: 
            Inpterpolated measurements for all location at one time
            A numpy array of shape (98, 5)
        """
        d = self.data_on_meshgrid(time)
        x_coord = self.lon.__array__()
        y_coord = self.lat.__array__()

        interpolators = [] # a list of interpolators, one for each fhour
        for fhour in range(len(self.fhour)):
            interpolators.append(interp2d(x_coord, y_coord, d[:,:,fhour], kind="linear"))
    
        all_stations = []
        for sn in self.station_names:
            x_coord, y_coord = self.get_station_position(sn)
            interpolated_var = self._interp_one_station(
                interpolators = interpolators, x_coord=x_coord, y_coord=y_coord)
            all_stations.append(interpolated_var)

        return np.array(all_stations)
            



