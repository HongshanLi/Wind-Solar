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
        start = time.time() 
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



class MultiThreads():
    def __init__(self, data_loc="/homes/li108/Dataset/SolarEnergy"):
        self.data_loc = data_loc
        self.var_dict = {k:v for k, v in enumerate(os.listdir(self.data_loc))}


        # Use a sample variable to initialize some attributes
        
        
        sample_var = os.listdir(data_loc)[0]
        sample_var = Dataset(os.path.join(data_loc, sample_var))
        self.time = sample_var.variables["time"]
        self.lat = sample_var.variables["lat"]
        self.lon = sample_var.variables["lon"]
        self.fhour = sample_var.variables["fhour"]

        
        self.all_features = range(len(self.var_dict))
        
        # Solar power station info
        self.station_info = pd.read_csv("station_info.csv")
        self.generated_energy = pd.read_csv("train.csv")
        self.station_names = self.station_info.columns[1:]


    def _load_one_variable(self, var_idx):
        """
        Takes the index of a variable
        Load <self.all_features[var_inx]> with one thread
        """
        var_path = os.path.join(self.data_loc, self.var_dict[var_idx])
        dataset = Dataset(var_path)
        values = dataset.variables.values()[-1]
        values = np.array(values)
        values = values.mean(axis=1)
        self.all_features[var_idx] = values



    def load_data(self):
        start = time.time()
        workers = []
        for idx in range(len(self.var_dict)):
            workers.append(Thread(target=self._load_one_variable(var_idx=idx)))
        
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        
        end = time.time()
        load_time = end - start
        print(load_time)
        
        
    


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
        lat, lon = self.get_station_position(station_name=station_name)
        x = self.interpolate_for_all_days(lat=lat, lon=lon, kind=kind)
        y = self.energy_generated_at_one_location(station_name)
        
        z = np.concatenate([x, y], axis=1)
        filename = station_name + r".npy"
        np.save(filename, z)

    def Main(self):
        for station_name in self.station_names:
            self.interpolated_data_for_one_location(station_name=station_name, kind="linear")



    


            

if __name__ =="__main__":
    w = WeatherData()
    w.Main()
    
        


        

