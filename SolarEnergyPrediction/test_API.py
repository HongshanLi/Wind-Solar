from scratch import OneVarInterp
import time
from Mesonet import WeatherData

# First check if the interpolated data makes sense
a = OneVarInterp()

start = time.time()
w = a.interp_once(time=0)
end = time.time()
print(end - start)
print(w.shape)



