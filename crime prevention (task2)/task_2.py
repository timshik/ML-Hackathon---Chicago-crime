import pandas as pd
from sklearn import cluster
from math import *
from datetime import datetime as dt

df = pd.read_csv('Dataset_crimes.csv', index_col=0)

class Cluster:
    def __init__(self):
        self.clstr_weekend = cluster.KMeans(n_clusters=30)
        self.clstr_midweek = cluster.KMeans(n_clusters=30)
        self.centers_weekend = []
        self.centers_midweek = []


    def fit(self, x_df_weekend, x_df_midweek):
        self.centers_weekend = self.clstr_weekend.fit_predict(x_df_weekend.dropna().to_numpy())
        self.centers_midweek = self.clstr_midweek.fit_predict(x_df_midweek.dropna().to_numpy())
        print (self.centers_midweek)



    def predict(self, date):
        date_format = dt.strptime(date , "%m/%d/%Y %I:%M:%S %p")
        day = date_format.weekday()
        if day >= 1 and day <= 5:
            return self.centers_midweek
        else:
            return self.centers_weekend


def hour_part(x):
    return int(x.hour/4)


def day_part(x):
    d = x.weekday()
    if d>=1 and d<=5:
        return 0
    return 1


def is_close_enough(lon1, lat1, lon2, lat2):
    Oppsite = 20000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    Base = 6371 * c
    distance = Base * 2 + Oppsite * 2 / 2
    return (distance / 1000) <= 500


def testing(x_test, clstr):
    countter = 0
    prediction = clstr.predict(x_test["DD-MM-YY"][0])
    for row in x_test:
        for loc_point in prediction:
            if is_close_enough(row["Longitude"],row["Latitude"], loc_point[0], loc_point[1]):
                counter = counter + 1
    return counter

def main():
    df = pd.read_csv("Dataset_crimes.csv")
    df.dropna()
    df["Date"] = df["Date"].apply(lambda x:dt.strptime(x,"%m/%d/%Y %I:%M:%S %p"))
    df["weekday"] = df["Date"].apply(day_part)
    df['YYYY-MM-DD'] = df['Date'].apply(lambda x:x.date())
    x_test = df[["Longitude", "Latitude","YYYY-MM-DD"]]
    x_test = x_test.drop(df[df["YYYY-MM-DD"] != dt.strptime("01/16/21", '%m/%d/%y').date()].index)
    df["hour"] = df["Date"].apply(hour_part)
    df = df[["hour","Longitude", "Latitude", "weekday"]]
    df_weekend = df[df["weekday"] == 1]
    df_weekend.drop(["weekday"], axis=1)
    df_midweek = df[df["weekday"] == 0]
    df_weekend.drop(["weekday"], axis=1)
    clstr = Cluster()
    clstr.fit(df_weekend, df_midweek)
    testing (x_test, clstr)


    print(df.head())



if __name__ == '__main__':
    main()
