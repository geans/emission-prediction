import json
import multiprocessing

import ordpy
import pandas as pd


class InformationHandleFile:
    counter = 1

    def __init__(self, path, window, dx, shift=1):
        self.__path = path
        self.__window = window
        self.__dx = dx
        self.__shift = shift
        self.__df_arr = multiprocessing.Manager().list()

    @staticmethod
    def get_sub_lists(original_list, delta):
        pivot = 0
        sub_lists = []
        len_list = len(original_list)
        shift = 1
        while pivot + delta <= len_list:
            sub_lists.append(original_list[pivot:pivot + delta])
            pivot += shift
        return sub_lists

    def get_parameters(self):
        return self.__dx

    @staticmethod
    def __run_df(df, window_size, dx, df_arr):
        car = df["Car_Id"].values[-1]
        person = df["Person_Id"].values[-1]
        trip = df["Trip"].values[-1]
        path_out = f'df-folder/{car}.{person}.{trip}.csv'
        InformationHandleFile.counter += 1
        if df.shape[0] < window_size:
            print('[!]', path_out, InformationHandleFile.counter)
            return
        else:
            print(path_out, InformationHandleFile.counter)
        sliding_window_df = InformationHandleFile.get_sub_lists(df, window_size)
        new_df = None
        new_df_sz = 0
        for window_df in sliding_window_df:
            row = {}
            for feature in window_df.columns:
                row[feature] = window_df[feature].values[-1]
                if 'OBD' in feature:
                    window_without_duplicate = window_df[feature].loc[window_df[feature].shift() != window_df[feature]]
                    if len(window_without_duplicate) < dx:
                        h = c = f = s = 'NaN'
                    else:
                        h, c = ordpy.complexity_entropy(window_without_duplicate, dx=dx)
                        s, f = ordpy.fisher_shannon(window_without_duplicate, dx=dx)
                    row[f'{feature}_entropy'] = h
                    row[f'{feature}_complexity'] = c
                    row[f'{feature}_shannon'] = s
                    row[f'{feature}_fisher'] = f
            if new_df is None:
                new_df = pd.DataFrame([row])
            else:
                new_df.loc[new_df_sz] = row
            new_df_sz += 1
        new_df.to_csv(path_out, index=False)
        df_arr.append(new_df)

    def __process_df(self, df):
        p = multiprocessing.Process(target=self.__run_df,
                                    args=(df, self.__window, self.__dx, self.__df_arr))
        p.start()
        return p

    def create_inf_measures_dataset(self):
        thread_pool = []
        df = pd.read_csv(self.__path, low_memory=False)
        for _, df_car in df.groupby('Car_Id'):
            for __, df_car_person in df_car.groupby('Person_Id'):
                for ___, df_car_person_trip in df_car_person.groupby('Trip'):
                    print(df_car_person_trip.shape)
                    thread = self.__process_df(df_car_person_trip)
                    thread_pool.append(thread)

        len_pool = len(thread_pool)
        print(f'{len_pool} threads. Window={self.__window}. dx={self.__dx}')

        for i, thread in enumerate(thread_pool, start=1):
            thread.join()
            print(f'{i}/{len_pool}')

        return list(self.__df_arr)


if __name__ == '__main__':
    path_in = 'VehicularData(anonymized).csv'
    path_out = 'VehicularData(anonymized)_inf.csv'

    handle = InformationHandleFile(path=path_in, window=300, dx=6, shift=1)
    df_arr = handle.create_inf_measures_dataset()
    print('counter =', InformationHandleFile.counter)
    with open('information.json', 'w') as out:
        json.dump(df_arr, out)
    df_processed = pd.concat(df_arr)
    df_processed.to_csv(path_out, index=False)
