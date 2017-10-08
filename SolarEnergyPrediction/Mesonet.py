from netCDF4 import Dataset
import numpy as np
import os
from scipy.interpolate import interp2d
import pandas as pd
from threading import Thread
import time

class WeatherData():
    def __init__(self, data_loc="/homes/li108/Dataset/SolarEnergy"):
        # Use a sample variable to initialize some attributes
        sample_var = os.listdir(data_loc)[0]
        sample_var = Dataset(os.path.join(data_loc, sample_var))
        self.time = sample_var.variables["time"]
        self.lat = sample_var.variables["lat"]
        self.lon = sample_var.variables["lon"]
        self.fhour = sample_var.variables["fhour"]

        # load all nc files
        """
        self.all_features = []
        for var in os.listdir(data_loc):
            var_path = os.path.join(data_loc, var)
            dataset = Dataset(var_path)
            values = dataset.variables.values()[-1]
            values = np.array(values)
            values = values.mean(axis=1) # take mean of ensemble of models
            self.all_features.append(values)
        
        end = time.time()
        load_time = end - start
        print(load_time)
        """

        # Solar power station info
        self.station_info = pd.read_csv("station_info.csv")
        self.generated_energy = pd.read_csv("train.csv")
        self.station_names = self.station_info.columns[1:]


    def _check_index(self, lat, lon):
        lat_idx = np.where(self.lat.__array__() == lat)
        lon_idx = np.where(self.lon.__array__() == lon)
        return lat_idx, lon_idx
        
    def _get_single_variable(self, var, time, lat, lon):
        lat_idx, lon_idx = self._check_index(lat, lon)
        return var[time,:,lat_idx,lon_idx].flatten()
    
    def _get_all_variables(self, time, lat, lon):
        all_vars = []
        for var in self.all_features:
            all_vars.append(self._get_single_variable(var, time,lat, lon))
        return np.concatenate(all_vars, axis=0).transpose()

       
    def data_on_meshgrid(self, var, time):
        """
        Arguments:
            var: a variable
            time: time index
        Return:
            The variable parsed to <var> on a meshgrid
            All 5 measurements are included
        """
        meshgrid_data = []
        for lat in self.lat.__array__():
            along_lat = []
            for lon in self.lon.__array__():
                along_lat.append(self._get_single_variable(var, time, lat, lon))
            meshgrid_data.append(along_lat)
        return np.array(meshgrid_data)

    def interpolate_single_variable(self, var, time, lat, lon, kind):
        """
        Arguments:
            var: a variable
            time: time index
            lat, lon : lattitude and longitude where you want to interpolate
                the variable <var>
            kind: the kind of interpolation ('linear', 'quadratic', 'cubic')
        Return:
            The interpolated variable at the location specifed by <lon> and <lat>
            All 5 measurements are included
        """
        d = self.data_on_meshgrid(var, time)
        x_coord = self.lon.__array__()
        y_coord = self.lat.__array__()

        
        interpolator = [] # a list of interpolator for each fhour
        for fhour in range(5):
            interpolator.append(interp2d(x_coord, y_coord, d[:,:,fhour], kind=kind))

        interpolated_data = []
        for fhour in range(5):
            interpolated_data.append(interpolator[fhour](lon, lat))
        
        return np.array(interpolated_data).flatten()

    def interpolate_for_one_day(self, time, lat, lon, kind):
        """
        This function interpolates all variables with all 5 measurement
        at the specified location.
        
        Arguments:
            time : time index
            lat, lon : lattitude and longitude where you want to interpolate
                all variables
        Return: 
            All variables interpolated at <lat, lon> at the <time>
            with all 5 measurements
        """
        interpolated_variables = []
        for var in self.all_features:
            interpolated_variables.append(
                self.interpolate_single_variable(var, time, lat, lon, kind))
        
        x = np.array(interpolated_variables)
        x = x.transpose()
        x = x.reshape(1, -1)
        return x

    def interpolate_for_all_days(self, lat, lon, kind):
        """
        This function interpolates all variable at <lat, lon> 
        for all 5113 days
        """
        all_data = []
        for i in range(len(self.time)):
            x = self.interpolate_for_one_day(time=i, lat=lat, lon=lon, kind=kind)
            all_data.append(x)

        return np.concatenate(all_data) 
        

    def get_station_position(self, station_name):
        """
        Arguments:
           station_name: The name of a solar power station
        Return:
            lattitude, longitude of that station
        """
        b = self.station_info
        lat = np.asscalar(b.loc[b['stid']==station_name]["nlat"])
        lon = np.asscalar(b.loc[b['stid']==station_name]["elon"])
        return lat, lon


    def energy_generated_at_one_location(self, station_name):
        """
        Arguments:
            station_name: The name of a solar power station
        Return:
            Array of energy generated at that location for all 5113 days
            Array shape = (5113, 1) to make concatenation easier
        """
        a = self.generated_energy[station_name]
        a = a.__array__()
        a = a.reshape(-1, 1)
        return a
        
    def interpolated_data_for_one_location(self, station_name, kind):
        """
        This method merge the interpolated inputs and label
        to a big array
        """
        start = time.time()

        lat, lon = self.get_station_position(station_name=station_name)
        x = self.interpolate_for_all_days(lat=lat, lon=lon, kind=kind)
        y = self.energy_generated_at_one_location(station_name)
        
        z = np.concatenate([x, y], axis=1)
        filename = station_name + r".npy"
        np.save(filename, z)
    
        end = time.time()
        proc_time = end - start
        print("Time to process one location is %f" % proc_time)
        

    def Main(self):
        for station_name in self.station_names:
            self.interpolated_data_for_one_location(station_name=station_name, kind="linear")




class WeatherDataMT():
    def __init__(self, data_loc="/homes/li108/Dataset/SolarEnergy"):
        # Use a sample variable to initialize some attributes
        sample_var = os.listdir(data_loc)[0]
        sample_var = Dataset(os.path.join(data_loc, sample_var))
        self.time = sample_var.variables["time"]
        self.lat = sample_var.variables["lat"]
        self.lon = sample_var.variables["lon"]
        self.fhour = sample_var.variables["fhour"]

        self.all_features = []
        """
        for var in os.listdir(data_loc):
            var_path = os.path.join(data_loc, var)
            dataset = Dataset(var_path)
            values = dataset.variables.values()[-1]
            values = np.array(values)
            values = values.mean(axis=1) # take mean of ensemble of models
            self.all_features.append(values)
        """
        
        self.interpolated_measurements = range(len(self.fhour))

        # Solar power station info
        self.station_info = pd.read_csv("station_info.csv")
        self.generated_energy = pd.read_csv("train.csv")
        self.station_names = self.station_info.columns[1:]


    def _check_index(self, lat, lon):
        lat_idx = np.where(self.lat.__array__() == lat)
        lon_idx = np.where(self.lon.__array__() == lon)
        return lat_idx, lon_idx
        
    def _get_single_variable(self, var, time, lat, lon):
        lat_idx, lon_idx = self._check_index(lat, lon)
        return var[time,:,lat_idx,lon_idx].flatten()
    
    def _get_all_variables(self, time, lat, lon):
        all_vars = []
        for var in self.all_features:
            all_vars.append(self._get_single_variable(var, time,lat, lon))
        return np.concatenate(all_vars, axis=0).transpose()

       
    def data_on_meshgrid(self, var, time):
        """
        Arguments:
            var: a variable
            time: time index
        Return:
            The variable parsed to <var> on a meshgrid
            All 5 measurements are included
        """
        meshgrid_data = []
        for lat in self.lat.__array__():
            along_lat = []
            for lon in self.lon.__array__():
                along_lat.append(self._get_single_variable(var, time, lat, lon))
            meshgrid_data.append(along_lat)
        return np.array(meshgrid_data)

    def _interpolate_one_measurement(self, fhour, var, time, lat, lon, kind):
        d = self.data_on_meshgrid(var, time)
        x_coord = self.lon.__array__()
        y_coord = self.lat.__array__()
        interpolator = interp2d(x_coord, y_coord, d[:,:,fhour], kind=kind)
        
        self.interpolated_measurements[fhour] = interpolator(lon, lat)


    def interpolate_single_variable(self, var, time, lat, lon, kind):
        """
        Arguments:
            var: a variable
            time: time index
            lat, lon : lattitude and longitude where you want to interpolate
                the variable <var>
            kind: the kind of interpolation ('linear', 'quadratic', 'cubic')
        Return:
            The interpolated variable at the location specifed by <lon> and <lat>
            All 5 measurements are included
        """
        interpolators = []
        for i in range(len(self.fhour)):
            interpolators.append(Thread(
                target=self._interpolate_one_measurement(fhour=i, var=var, time=time,
                    lat=lat, lon=lon, kind=kind)))

        for interpolator in interpolators:
            interpolator.start()
        return np.array(self.interpolated_measurements).flatten()


        



    def interpolate_for_one_day(self, time, lat, lon, kind):
        """
        This function interpolates all variables with all 5 measurement
        at the specified location.
        
        Arguments:
            time : time index
            lat, lon : lattitude and longitude where you want to interpolate
                all variables
        Return: 
            All variables interpolated at <lat, lon> at the <time>
            with all 5 measurements
        """
        interpolated_variables = []
        for var in self.all_features:
            interpolated_variables.append(
                self.interpolate_single_variable(var, time, lat, lon, kind))
        
        x = np.array(interpolated_variables)
        x = x.transpose()
        x = x.reshape(1, -1)
        return x

    def interpolate_for_all_days(self, lat, lon, kind):
        """
        This function interpolates all variable at <lat, lon> 
        for all 5113 days
        """
        all_data = []
        for i in range(len(self.time)):
            x = self.interpolate_for_one_day(time=i, lat=lat, lon=lon, kind=kind)
            all_data.append(x)

        return np.concatenate(all_data) 
        

    def get_station_position(self, station_name):
        """
        Arguments:
           station_name: The name of a solar power station
        Return:
            lattitude, longitude of that station
        """
        b = self.station_info
        lat = np.asscalar(b.loc[b['stid']==station_name]["nlat"])
        lon = np.asscalar(b.loc[b['stid']==station_name]["elon"])
        return lat, lon


    def energy_generated_at_one_location(self, station_name):
        """
        Arguments:
            station_name: The name of a solar power station
        Return:
            Array of energy generated at that location for all 5113 days
            Array shape = (5113, 1) to make concatenation easier
        """
        a = self.generated_energy[station_name]
        a = a.__array__()
        a = a.reshape(-1, 1)
        return a
        
    def interpolated_data_for_one_location(self, station_name, kind):
        """
        This method merge the interpolated inputs and label
        to a big array
        """
        start = time.time()

        lat, lon = self.get_station_position(station_name=station_name)
        x = self.interpolate_for_all_days(lat=lat, lon=lon, kind=kind)
        y = self.energy_generated_at_one_location(station_name)
        
        z = np.concatenate([x, y], axis=1)
        filename = station_name + r".npy"
        np.save(filename, z)
    
        end = time.time()
        proc_time = end - start
        print("Time to process one location is %f" % proc_time)
        

    def Main(self):
        for station_name in self.station_names:
            self.interpolated_data_for_one_location(station_name=station_name, kind="linear")




    
        


        

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
            



