from netCDF4 import Dataset
import numpy as np
import os

class WeatherData():
    def __init__(self, data_loc):
        # Use a sample variable to initialize some attributes
        sample_var = os.listidr(data_loc)[0]
        sample_var = Dataset(os.path.join(data_loc, sample_var))
        time = 

        # load all nc files
        self.all_variables = []
        for var in os.listdir():
            var_path = os.path.join(data_loc, var)
            dataset = Dataset(var_path)
            values = dataset.variables.values
        

    def _check_index()


    def location(self, lat, lon):
        # Arguments: actual values of lattitude and longitude (not index)
        # 
        
        return (lat, lon), data_at_this_location

