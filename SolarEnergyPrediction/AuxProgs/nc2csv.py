from netCDF4 import Dataset
import os

data_loc = "/home/hongshan/Dataset/SolarEnergy"
var1 = os.listdir(data_loc)[0]


dataset = Dataset(os.path.join(data_loc, var1))
print(dataset.file_format)

print(dataset["lat"][0:5])
lats = dataset.variables["lat"][:]
print(lats)

variables = dataset.dimensions.keys()
for var in variables:
    print(var, len(dataset.variables[var][:]))

# def get_variable(which_var, time, lat, lon, ens, fhour):
    
