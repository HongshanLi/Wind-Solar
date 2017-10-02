from Mesonet import WeatherData

a = WeatherData()
b = a._get_all_variables(time=0, lat=32, lon=254)
print(b)
print(b.shape)
