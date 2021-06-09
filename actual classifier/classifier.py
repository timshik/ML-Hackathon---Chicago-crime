import pickle
import numpy as np
from datetime import datetime as dt
import pandas as pd


class Classifier:
    def init(self):
        pass

    def fit(self, x_df, y_df):
        pass

    def predict(self, x_def):
        pass

    def score(self, x_df, y_df):
        pass


def make_categorical(feature, x_df):
    feature_column = pd.get_dummies(x_df[feature], prefix=feature, drop_first=True)
    x_df.drop(feature, axis='columns', inplace=True)
    x_df = pd.concat([x_df, feature_column], axis=1)
    return x_df


def date_pre(df):
    df_date = df["Date"].apply(lambda x: dt.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
    df_update = df["Updated On"].apply(lambda x: dt.strptime(x, "%m/%d/%Y %I:%M:%S %p"))

    df["month"] = df_date.apply(lambda x: x.month)
    # df["Mday"]=df_date.apply(lambda x:x.day)
    df["Wday"] = df_date.apply(lambda x: x.weekday())
    df["hour"] = df_date.apply(lambda x: x.hour)  # +x.minute/60)
    df["days from update"] = (df_update - df_date).apply(lambda x: x.days)
    df.drop("Date", axis='columns', inplace=True)
    df.drop("Updated On", axis='columns', inplace=True)
    return df


def dimension_reduction(df, feature):
    counter = df[feature].value_counts()
    feature_list = counter.index.tolist()

    # count number of cases for each location and type of crime
    df_1 = df[[feature, "Primary Type"]]
    grouped_df = df_1.groupby([feature, "Primary Type"]).size().reset_index()
    grouped_df = grouped_df.rename(columns={0: "count"})
    important_values = []
    for val in grouped_df[feature].unique():
        if grouped_df[grouped_df[feature] == val].max().to_numpy()[2] / \
                grouped_df[grouped_df[feature] == 111].sum().to_numpy()[2] > 0.5:
            important_values.append(val)
    return important_values


def filter_feature_values(val, lst_values):
    if not val in lst_values:
        return 0
    else:
        return val


def PreProcess_dataset(dataset, list_of_features, list_of_categorical_features):
    dataset = date_pre(dataset)
    dataset = pre_location(dataset)
    # dataset = pre_feature(dataset, 'Beat')
    #
    x_df = dataset[list_of_features + list_of_categorical_features]
    for feature in list_of_categorical_features:
        x_df = make_categorical(feature, x_df)

    return x_df


def data_processing(path, list_of_features, list_of_categorical_features):
    df = pd.read_csv(path)
    df = df.drop(["Unnamed: 0"], axis=1)
    df = df.dropna()
    # split train & test
    y = df["Primary Type"]
    X = df.drop(["Primary Type"], axis=1)
    X = PreProcess_dataset(X, list_of_features, list_of_categorical_features)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X
    # return X_train, y_train, X_test, y_test


def second_pre(df):
    df_date = df["Date"].apply(lambda x: dt.strptime(x, "%m/%d/%Y %H:%M:%S %p"))
    df["Wday"] = df_date.apply(lambda x: x.weekday())
    df["hour"] = df_date.apply(lambda x: x.hour + (0.5 if x.minute >= 30 else 0))
    ndf = df[["Wday", "hour", "Y Coordinate", "X Coordinate"]]
    return ndf


def rename_location(location, lst_locations):
    if not location in lst_locations:
        return "other"
    if "store" in location.lower():
        return "store"
    else:
        return location


def pre_feature(df, feature):
    important_values = dimension_reduction(df, feature)
    # replace nan values
    location_df = df[[feature]].replace(np.nan, 'other')
    # get lost of location where there were big number of cases
    counter = df[feature].value_counts()
    location_others = counter.index.tolist()
    # convert all different "store" locations to store
    location_df = location_df.applymap(lambda x: filter_feature_values(x, important_values))
    location_df[feature].unique()
    # replace the old location column
    df[feature] = location_df[feature]
    return df


def pre_location(df):
    # replace nan values
    location_df = df[['Location Description']].replace(np.nan, 'other')
    # get lost of location where there were big number of cases
    counter = df['Location Description'].value_counts()
    counter = counter[counter > 100]
    location_others = counter.index.tolist()
    # convert all different "store" locations to store
    location_df = location_df.applymap(lambda x: rename_location(x, location_others))
    location_df["Location Description"].unique()
    # replace the old location column
    df["Location Description"] = location_df["Location Description"]
    return df


def classify(dataset_path):
    crimes_dict = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
    list_of_features = ['Arrest', 'Domestic', "days from update"]
    list_of_categorical_features = ['Beat', 'Ward', 'Community Area', 'District', 'Location Description', "month",
                                    "Wday", "hour"]
    X = data_processing(dataset_path, list_of_features, list_of_categorical_features)
    with open('RandomForestModel.pickle', 'rb') as handle:
        clf = pickle.load(handle)
    predictions = clf.predict(X)
    y_pred = [crimes_dict[pred] for pred in predictions]
    return y_pred







#second meission
def predict(X):
    pass
def day_part(x):
    d=x.weekday()
    if d>=2 and d<=5:
        return 0
    return 1
def send_police_cars(t):
    allmap = np.load("allmap.npy")
    with open("lockmap", "rb") as fp:  # Unpickling
        locmap = pickle.load(fp)

    t = dt.strptime(t, "%m/%d/%Y %I:%M:%S %p")
    da = (1 if day_part(t) == "reg" else 0)
    am = allmap[da]
    l = np.zeros(40).tolist()
    for i in range(len(l)):
        l[i] = np.unravel_index(np.argmax(am, axis=None), am.shape)
        am[l[i]] = 0

    out = np.zeros(40).tolist()
    i = 0
    for a in l:
        cur = pd.DataFrame(locmap[da][a[0]][a[1]][a[2]])
        if float(cur["X Coordinate"].median()) == 0 or float(cur["Y Coordinate"].median()) == 0:
            continue

        out[i] = (cur["X Coordinate"].median(), cur["Y Coordinate"].median(),
                  dt(t.year, t.month, t.day, int(cur["hour"].median()), int((cur["hour"].mean() % 1) * 60), 0))
        i += 1
    return out[0:30]

