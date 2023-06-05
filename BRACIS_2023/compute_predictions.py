from predict_ByYear import PredictByYear

# Setando entrada
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCLSTM'
reps = 10

predict_1 = PredictByYear(lags=lags, datasets=datasets, weight=weight, out=out, nameModel=nameModel, reps=reps)
prediction_1 = predict_1.predict()

# Setando entrada
lags = 14
datasets = '2020-2022'
weight = 'all'
out = 'cases'
nameModel = 'GCRN'
reps = 10

predict_2 = PredictByYear(lags=lags, datasets=datasets, weight=weight, out=out, nameModel=nameModel, reps=reps)
prediction_2 = predict_2.predict()
