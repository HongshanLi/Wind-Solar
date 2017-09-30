from netCDF4 import Dataset
import numpy as np
import os
from scipy import interpolate
import matplotlib.pyplot as plt


class WeatherData():
    def __init__(self, data_loc="/home/hongshan/Dataset/SolarEnergy"):
        # Use a sample variable to initialize some attributes
        sample_var = os.listdir(data_loc)[0]
        sample_var = Dataset(os.path.join(data_loc, sample_var))
        self.time = sample_var.variables["time"]
        self.lat = sample_var.variables["lat"]
        self.lon = sample_var.variables["lon"]
        self.fhour = sample_var.variables["fhour"]

        # load all nc files
        self.all_features = []
        for var in os.listdir(data_loc):
            var_path = os.path.join(data_loc, var)
            dataset = Dataset(var_path)
            values = dataset.variables.values()[-1]
            values = np.array(values)
            values = values.mean(axis=1) # take mean of ensemble of models
            self.all_features.append(values)
        

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

    def _make_grid(self):
        grid_x = np.concatenate([self.lon.__array__()]*len(self.lat.__array__()))
        y = self.lat.__array__()
        y = y.reshape(1, -1)
        grid_y = np.concatenate([y]*len(self.lon.__array__()), axis=0)
        grid_y = grid_y.flatten("F")
        plt.scatter(x=grid_x, y=grid_y, color="b")
        plt.show()

    def interpolate(self, time, lat, lon, kind):
        x_coord = self.lon.__array__()
        y_coord = self.lat.__array__()
        xx, yy = np.meshgrid(x_coord, y_coord)
        # I need 2d array of data here
        


if __name__ =="__main__":
    w = WeatherData()
    print(w._get_single_variable(w.all_features[0], 31, 260).shape)
    
        


        

