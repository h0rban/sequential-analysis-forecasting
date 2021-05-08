import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tqdm import tqdm
from datetime import datetime
from scipy.optimize import leastsq
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score as r2


class Model:
    def __init__(self,
                 dates: np.array,
                 estimator=None,
                 estimator_fn=None,
                 M: int = 120,
                 vis_build: bool = False,
                 vis_predict: bool = False,
                 write_imgs: bool = False,
                 **kwargs):
        """
        Model constructor
        :param estimator_fn: estimator function
        :param estimator: estimator
        :param dates: list of dates
        :param M: number of days for sequential scanning
        :param vis_build: should the build be visualized
        :param vis_predict: should the prediction be visualized
        :param write_imgs: should the visualization be written
        :param kwargs: estimator parameters
        """

        if estimator is None and estimator_fn is None:
            raise Exception("either estimator or estimator fn has to be provided")

        self.M = M
        self.dates = dates
        self.vis_build = vis_build
        self.vis_predict = vis_predict
        self.write_imgs = write_imgs
        self.model = MultiOutputRegressor(estimator if estimator_fn is None else estimator_fn(**kwargs))
        self.y_train = None
        self.log_params = None
        self.train_size = None
        self.train_pred = None
        self.train_score = None
        self.initial_params = None
        self.sin_param_score = None

    @staticmethod
    def __sin(x, params: np.array):
        """
        f(x) = a * sin(bx + c) + d
        :param x: x
        :param params: sinusoidal parameter
        :return: result of f(x)
        """
        assert params.size == 4
        a, b, c, d = params
        return a * np.sin(b * x + c) + d

    @staticmethod
    def __log(x, params: np.array):
        """
        f(x) = a / (1 + e^(bx)) + d
        :param x: x
        :param params: b parameter of the sigmoid
        :return: result of f(x)
        """
        assert params.size == 1
        # a, b, d = params
        a, b, d = 1, params[0], -.5
        return a / (1 + np.exp(b * x)) + d

    def __log_sin(self, x, params: np.array):
        """
        f(x) = a * sin(bx + c) + d
        g(x) = a / (1 + e^(bx)) + d
        l(x) = g(x) * f(x)
        :param x: x
        :param params: parameters of sinusoid and sigmoid
        :return: result of l(x)
        """
        assert params.size == 5
        return self.__log(x, params[0:1]) * self.__sin(x, params[1:])

    def __log_plus_sin(self, x, params: np.array):
        """
        f(x) = a * sin(bx + c) + d
        g(x) = a / (1 + e^(bx)) + d
        l(x) = g(x) + f(x)
        :param x: x
        :param params: parameters of sinusoid and sigmoid
        :return: result of l(x)
        """
        assert params.size == 5
        return self.__log(x, params[0:1]) + self.__sin(x, params[1:])

    @staticmethod
    def __optimize(x, params, func, y_true):
        """
        Optimizes the parameters of the given function to best fit the true values
        :param x: x
        :param params: parameters of func
        :param func: function to optimize
        :param y_true: true values
        :return: optimal parameters, func(x) with the optimal values
        """
        params_opt = leastsq(lambda var: (func(x, var) - y_true) ** 2, params)[0]
        data_opt = func(x, params_opt)
        return params_opt, data_opt

    @staticmethod
    def __get_param_estimate(data):
        """
        Returns the initial estimate of the parameters given data
        :param data: data
        :return: parameters initial estimate
        """
        sigmoid_rate = -.01  # todo could estimate from data
        sin_a = data.mean() / 4
        sin_b = 2 * np.pi / (558 - 507)  # todo find this by finding two maximums for period
        sin_c = 0
        sin_d = 0
        return np.array([sigmoid_rate, sin_a, sin_b, sin_c, sin_d])

    def __sin1(self, x, params: np.array):
        assert self.log_params is not None
        return self.__log(x, self.log_params) + self.__sin(x, params)

    @staticmethod
    def __sin2(x, params: np.array):
        # f(x) = a * sin(bx + c) + d
        assert params.T.shape[0] == 4
        a, b, c, d = params.T
        return a * np.sin(b * x + c) + d

    def __log_sin_2(self, x, params):
        return self.__log(x, self.log_params) + self.__sin2(x, params)

    @staticmethod
    def make_param_dict(params):
        """
        build sequential sin parameters
        :param params: sin params
        :return: dictionary of
        """
        param_names = ['sin_a', 'sin_b', 'sin_c', 'sin_d']
        assert len(params) == len(param_names)
        dic = dict()
        for i in range(len(params)):
            dic[param_names[i]] = params[i]
        return dic

    def vis_prediction(self, test_pred, y_test):
        # build visualization
        n = len(self.dates)
        full_y_train = np.full(n, None)
        full_y_test = np.full(n, None)
        full_est_train = np.full(n, None)
        full_est_test = np.full(n, None)

        x = np.arange(self.train_size)
        full_y_train[x] = self.y_train
        full_est_train[x] = self.train_pred

        x = np.arange(self.train_size, n)
        full_y_test[x] = y_test
        full_est_test[x] = test_pred

        fig = px.line(pd.DataFrame({
            'y_train': full_y_train,
            'y_test': full_y_test,
            'est_train': full_est_train,
            'est_test': full_est_test
        }, index=self.dates),
            title='Prediction Visualization',
            labels={'index': 'date'},
            color_discrete_map={
                'y_train': 'royalblue',
                'y_test': 'royalblue',
                'est_train': '#00CC96',
                'est_test': '#EF553B'
            },
            render_mode='svg')

        if self.vis_predict:
            print(pd.Series([
                self.sin_param_score,
                self.train_score,
                r2(y_test, test_pred)
            ], index=[
                'sin_params',
                'train',
                'test'
            ], name=f'R2 of {self.model.estimators_[0]}').round(3))
            fig.show()
        if self.write_imgs:
            fig.write_image('images/prediction.svg')

    def show_write_fig(self, path, fig):
        if self.vis_build:
            fig.show()
        if self.write_imgs:
            fig.write_image(f'images/{path}')

    def write_seq_fitting(self, i, n, x, y, data_opt):
        if self.write_imgs and i % 10 == 0:
            est = np.full(n, None)
            est[x] = data_opt
            index = self.training_dates()
            fig = go.Figure() \
                .add_trace(go.Scatter(x=index, y=y, name='y_train')) \
                .add_trace(go.Scatter(x=index, y=est, name='estimate'))
            fig.write_image(f'images/frames/frame_{i}.png')

            if i == 500:  # todo dont hardcode
                fig.update_layout(title=f'Fitting at i = {i}')
                fig.update_xaxes(showticklabels=False)
                fig.write_image(f'images/fitting.svg')

    def estimators(self):
        return self.make_param_dict(self.model.estimators_)

    def training_dates(self):
        assert self.train_size is not None
        return self.dates[:self.train_size]

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        assert len(self.dates) > n
        self.train_size = n

        # fit a log + sin to y
        first_guess = self.__get_param_estimate(y)
        x = np.arange(n)
        params_opt, data_opt = self.__optimize(x, first_guess, self.__log_plus_sin, y)

        # save the initial estimates
        self.initial_params = params_opt
        self.log_params = params_opt[0:1]
        sin_params = params_opt[1:]

        # visualize initial estimate
        if self.vis_build:
            print(f'log_param: {self.log_params}\nsin_params: {sin_params}')
            print(f'R2 = {r2(y, data_opt)}')

        if self.write_imgs or self.vis_build:
            self.show_write_fig('initial.svg',
                                px.line(pd.DataFrame({'y_train': y, 'estimate': data_opt},
                                                     index=self.dates[:n]),
                                        title='Initial estimate',
                                        labels={'index': 'date'},
                                        render_mode='svg'))

        r2s = []
        x = np.arange(0, self.M)
        params_opt, data_opt = self.__optimize(x, sin_params, self.__sin1, y[x])
        prev_params = params_opt
        sin_params = []  # list to build
        i_range = tqdm(range(n)) if self.vis_build else range(n)
        for i in i_range:
            if i >= self.M:  # fill first M values to the first
                x = np.arange(i + 1 - self.M, i + 1)
                params_opt, data_opt = self.__optimize(x, prev_params, self.__sin1, y[x])
                prev_params = params_opt

                self.write_seq_fitting(i, n, x, y, data_opt)

            # save parameters
            r2s.append(r2(y[x], data_opt))
            sin_params.append(self.make_param_dict(params_opt))

        sin_params = pd.DataFrame(sin_params, index=self.training_dates())
        r2s = np.array(r2s)

        if self.write_imgs or self.vis_build:
            self.show_write_fig('sin_params.svg',
                                px.line(sin_params,
                                        title='Sequential parameters of the sinusoid',
                                        labels={'index': 'date'},
                                        render_mode='svg'))

            self.show_write_fig('r2s.svg',
                                px.line(pd.DataFrame({'y_train': y, 'r2': r2s},
                                                     index=self.training_dates()),
                                        title='R2 of the sequential fitting',
                                        labels={'index': 'date'},
                                        render_mode='svg'))

        self.model.fit(X, sin_params)
        # self.model.fit(X, sin_params, sample_weight=r2s)
        pred_sin_params = self.model.predict(X)
        self.train_pred = self.__log_sin_2(np.arange(n), pred_sin_params)
        self.sin_param_score = self.model.score(X, sin_params)
        self.train_score = r2(y, self.train_pred)
        self.y_train = y

        return self

    def predict(self, X, y):
        n, d = X.shape

        # todo it might be a good idea to use the initial estimate if the regressor does not work
        if n == self.train_size:
            return self.train_pred
        elif n == len(self.dates) - self.train_size:

            x = np.arange(self.train_size, len(self.dates))
            test_pred = self.__log_sin_2(x, self.model.predict(X))
            self.vis_prediction(test_pred, y)

            # sanity check - prediction using initial estimate
            # test_pred = self.__log_plus_sin(x, self.initial_params)
            # self.vis_prediction(test_pred, y)
            # print(r2(y, test_pred))

            return test_pred

        raise Exception("unexpected size of X")

    def score(self, X, y, all_three: bool = False):
        n, d = X.shape

        # todo check the size to be either train or test
        if n == self.train_size:
            return self.train_score
        elif n == len(self.dates) - self.train_size:
            x = np.arange(self.train_size, len(self.dates))
            test_pred = self.__log_sin_2(x, self.model.predict(X))

            test_score = r2(y, test_pred)

            if all_three:
                return pd.Series([
                    self.sin_param_score,
                    self.train_score,
                    test_score
                ], index=[
                    'sin_params',
                    'train',
                    'test'
                ]).round(3)

            return test_score

        raise Exception("unexpected size of X")


def train_test_scaled_split(df: pd.DataFrame, p: float = .20, include_x: bool = True,
                            scaler=MinMaxScaler()) -> tuple:
    assert 0 < p < 1

    # determine the number of rows that will go to testing
    length = len(df)
    M = int(length // (1 / p) + 1)
    n = length - M

    cols_to_scale = [item for item in df.columns if item not in ['date', 'imp_day']]

    train2scale = df[cols_to_scale].iloc[range(n), :]
    test2scale = df[cols_to_scale].iloc[range(n, length), :]

    # save train and test dates for future use
    dates = np.array(df['date'])

    # scale train and test data
    scaler.fit(train2scale)
    train = scaler.transform(train2scale)
    test = scaler.transform(test2scale)

    # all but last column (target)
    X_train = train[:, :-1]
    X_test = test[:, :-1]

    # last column is target
    y_train = train[:, -1]
    y_test = test[:, -1]

    # append important date
    imp_day_train = np.array(df['imp_day'][range(n)]).reshape((-1, 1))
    imp_day_test = np.array(df['imp_day'][range(n, length)]).reshape((-1, 1))

    X_train = np.concatenate([X_train, imp_day_train], axis=1)
    X_test = np.concatenate([X_test, imp_day_test], axis=1)

    # return columns if data were to be reconstructed into the data frame
    cols_to_scale.pop()  # remove target
    cols = cols_to_scale + ['imp_day']

    if include_x:
        x = np.arange(n).reshape((-1, 1))
        X_train = np.concatenate([x, X_train], axis=1)
        x = np.arange(n, length).reshape((-1, 1))
        X_test = np.concatenate([x, X_test], axis=1)
        cols = ['x'] + cols

    d = len(cols)

    assert X_train.shape == (n, d) and X_test.shape == (M, d)

    return X_train, y_train, X_test, y_test, cols, dates


def get_data(smooth: bool = False, set_index=False, only_keep_avg=True) -> pd.DataFrame:
    url_org_data = 'https://raw.githubusercontent.com/h0rban/flower-harvest-prediction/master/data/redeagle.csv'

    # convert col names to lower string, rename target variable
    df = pd.read_csv(url_org_data).rename(str.lower, axis='columns').rename(columns={'produced': 'target'})

    # remove columns that have '_min' and '_max' in their name
    if only_keep_avg:
        for name in df.columns:
            if '_min' in name or '_max' in name:
                df.drop(name, axis=1, inplace=True)

    # convert string date into datetime object
    df['date'] = df['date'].apply(lambda date: datetime.strptime(date, "%m/%d/%y"))

    # add important dates column
    def get_close_dates(observed: pd.Series, month_days: list) -> list:
        y_unique = list(observed.apply(lambda x: x.year).unique())
        y_unique.append(min(y_unique) - 1)
        y_unique.append(max(y_unique) + 1)
        y_unique.sort()
        close_dates = []
        for month, day in month_days:
            for year in y_unique:
                close_dates.append(datetime(year, month, day))
        return close_dates

    month_days = [(3, 8), (9, 1), (2, 14)]
    imp_dates = get_close_dates(df['date'], month_days)

    def n_days_before(date: datetime, imp_dates: list = imp_dates, n: int = 7):
        for d in imp_dates:
            dif = (d - date).days
            if 0 <= dif < n:
                return 1
        return 0

    df['imp_day'] = df['date'].apply(n_days_before)

    if set_index:
        df = df.set_index('date')

    if smooth:
        def moving_avg(s: pd.Series, M: int = 7, precision: int = 2) -> pd.Series:
            n = s.size
            arr = np.zeros(n)
            for i in range(n):
                if i == 0:
                    arr[i] = s[i]
                elif i < M:
                    arr[i] = round(s[0: i + 1].mean(), precision)
                else:
                    arr[i] = round(s[i - M: i + 1].mean(), precision)

            return pd.Series(arr, index=s.index, name=f'{s.name}_ma{M}')

        for col in set(df.columns) - {'imp_day', 'lamps', 'date'}:
            df[col] = moving_avg(df[col])

    return df
