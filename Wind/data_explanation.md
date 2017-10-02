# Data explanation

## raw data
download from source directly,

"train.csv" contains the training data:

- the first column ("date") is a timestamp giving date and time of the hourly wind power measurements in following columns. For instance "2009070812" is for the 8th of July 2009 at 12:00;

- the following 7 columns ("wp1" to "wp7") gather the normalized wind power measurements for the 7 wind farms. They are normalized so as to take values between 0 and 1 in order for the wind farms not to be recognizable.

"windforecasts_wf\*.csv" contains the wind forecasts for 7 wind farms:

- the first column ("date") is a timestamp giving date and time at which the forecasts are issued. For instance "2009070812" is for the 8th of July 2009 at 12:00;

- the second column ("hors") is for the lead time of the forecast. For instance if "date" = 2009070812 and "hors" = 1, the forecast is for the 8th of July 2009 at 13:00

- the following 4 columns ("u", "v", "ws" and "wd") are the forecasts themselves, the first two being the zonal and meridional wind components, while the following two are corresponding wind speed and direction.

"benchmark.csv" contains the benchmark data:

- Provide example forecast results from the persistence forecast method, *is not the wind power measurements for use as the test data*.

## pre-cleaned data

different clean methods can be applied to this dataset to get the cleaned data.

"train_wf\*.csv" data for 7 wind farm seperately:

- 1st column is the timestamp, 2nd-5th is the nearest prediction of the timestampi(), 6th-9th is the second nearest prediction,……, totally 4 predictions. the last column is the wind power measurement.

"benchmark\*.csv" data for 7 wind farm seperately:
- columns have the same meaning as the train_wf\*.csv.

## cleaned data

no header data.

"train_wf\*.csv" data for 7 wind farm seperately:
- 1st-4th columns are the nearest prediction of the first timestamp,……, the last column is the measurement. If the data has few than 4 predictions, such as the 1st timestamp data, then, the average of the nearest data are used as the missing data.

"benchmark\*.csv" data for 7 wind farm seperately:
- columns have the same meaning as the train_wf\*.csv.
