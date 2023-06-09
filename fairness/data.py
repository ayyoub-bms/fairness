import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import make_column_transformer


URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


def load_adult_data(onehot=False, labeling=True, clean_missing=True,
                    standardize=True):
    """ Loads adult data downloaded from UCI ML website
    Also known as "Census Income" dataset.

    Predict whether income exceeds $50K/yr based on census data.

    Parameters
    ----------

        onehot: bool
            Whether the categorical variables should be one hot encoded

        labeling: bool
            Transform strings into integer labels

        clean_missing: bool
            Remove missing data

        standardize: bool
            Transform data with mean > 1 to have 0 mean and 1 std.

    Returns
    -------

        tuple of (X_train, X_test, y_train, y_test, sensitive)

    See Also
    --------

        sklearn.model_selection.train_test_split
    """
    
    sensitive = 'gender'

    data_train = _get_data(URL + 'adult/', 'adult.data', header=None)
    data_tests = _get_data(URL + 'adult/', 'adult.test', header=None, skiprows=1)
    
    n = data_train.shape[0]
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income'
    ]

    # includes only features with more than 2 categories
    categorical = [
        'workclass', 'marital_status', 'occupation',
        'relationship', 'race', 'native_country',
    ]

    data_train.columns = columns
    data_tests.columns = columns

    data_tests['income'] = data_tests['income'].str.replace(
        ' >50K.', ' >50K', regex=False
    )
    data = pd.concat([data_train, data_tests])

    # drop missing values
    if clean_missing:
        # Missing data are reported as ' ?'
        data.replace(' ?', np.nan, regex=False, inplace=True)
        data.dropna(inplace=True)

    # set the sensitive feature
    data['gender'] = (data['gender'] == ' Female').astype(int)

    data['marital_status'] = data['marital_status'].str.replace(
        ' Married*', 'Married', regex=True
    )

    data['marital_status'] = data['marital_status'].str.replace(
        '^(?!Married).*', 'Not-Married', regex=True
    )

    # Building the target
    target = (data['income'] == ' >50K').astype(int)

    to_drop = ['education', 'income', 'fnlwgt']

    # We don't need the target variable in the features dataset
    data.drop(to_drop, axis=1, inplace=True)

    if labeling:
        encoder = LabelEncoder()
        for c in categorical:
            data[c] = encoder.fit_transform(data[c])

    if onehot:
        # One hot encoding setup
        # Transform each categorical data to its one hot counterpart
        data = _binarize(data, categorical)

    target_train = target.iloc[:n]
    target_tests = target.iloc[n:]

    data_train = data.iloc[:n]
    data_tests = data.iloc[n:]

    del data

    if standardize:
        if labeling:
            to_scale = data_train.columns.difference([sensitive])
        else:
            to_scale = list(
                data_train.columns.difference(
                    categorical + to_drop + [sensitive]
                )
            )
        
        scaler = StandardScaler()
        scaler.fit(data_train[to_scale])
        data_train[to_scale] = scaler.transform(data_train[to_scale])
        data_tests[to_scale] = scaler.transform(data_tests[to_scale])

    return data_train, target_train, data_tests, target_tests, sensitive


def load_drug_data(drug='heroin', train_size=.7, standardize=True, **kwargs):
    """ Load drug data downloaded from UCI ML website


    The database contains records for 1885 respondents and 12
    attributes.

    Parameters
    ----------

        drug: str
            One of the drug consumptions to classify amongst
            01- alcohol
            02- amphet
            03- amyl
            04- benzos
            05- caff
            06- cannabis
            07- choc
            08- coke
            09- crack
            10- ecstasy
            11- heroin
            12- ketamine
            13- legalh
            14- lsd
            15- meth
            16- mushrooms
            17- nicotine
            18- semer
            19- vsa

        train_size: float
            Percentage of train data for train/test split.

        standardize: bool
            whether the data should have 0 mean and 1 std.

        **kwargs: dict
            Any parameters to pass to the train test split function
            train_test_split

    Returns
    -------

        tuple of (X_train, X_test, y_train, y_test, sensitive)

    See Also
    --------

        sklearn.model_selection.train_test_split
    """
    sensitive = 'ethnicity'

    numerical = [
        'age', 'gender', 'education', 'country', 'ethnicity', 'n_score',
        'e_score', 'o_score', 'a_score', 'c_score', 'impulsive', 'ss'
    ]

    categorical = [
        'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc',
        'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd',
        'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'
    ]

    assert (drug in categorical), f'Unkown drug name {drug}'

    data = _get_data(URL + '00373/', 'drug_consumption.data', header=None)

    data.columns = ['ID'] + numerical + categorical

    numerical.remove(sensitive)

    data.set_index('ID', inplace=True)

    target = (data[drug] != 'CL0').astype(int)
    
    data = data[numerical + [sensitive]]
    data[sensitive] = (data[sensitive] == -0.31685).astype(int)

    data_train, target_train, data_tests, target_tests, sensitive = _gen_data(
        data, target, train_size, sensitive, **kwargs
    )

    del data
    
    if standardize:
        scaler = StandardScaler()
        scaler.fit(data_train[numerical])
        data_train[numerical] = scaler.transform(data_train[numerical])
        data_tests[numerical] = scaler.transform(data_tests[numerical])

    return data_train, target_train, data_tests, target_tests, sensitive


def load_german_data(onehot=True, labeling=True, standardize=True,
                     train_size=.7, **kwargs):
    """ Loads German credit data downloaded from UCI ML website

    This dataset classifies people described by a set of attributes as good or
    bad credit risks.

    Parameters
    ----------

        onehot: bool
            Whether the categorical variables should be one hot encoded

        labeling: bool
            Whether the categorical variables should be labeled

        standardize: bool
            Transform data with max > 1 to have 0 mean and 1 std.

        train_size: float
            Percentage of train data for train/test split.

        **kwargs: dict
            Any parameters to pass to the train test split function
            train_test_split

    Returns
    -------

        tuple of (X_train, X_test, y_train, y_test, sensitive)

    See Also
    --------

        sklearn.model_selection.train_test_split
    """

    sensitive = 'foreign'

    data = _get_data(URL + 'statlog/german/', 'german.data',
                     header=None,
                     sep='\\s+')

    numerical = [
        'duration', 'crdtamt', 'installment', 'residsince', 'age', 'nbcred',
        'nbliable'
    ]

    data.columns = [
        'accstatus', 'duration', 'history', 'purpose', 'crdtamt',
        'savings', 'employment', 'installment', 'maritalgender',
        'debtgarant', 'residsince', 'property', 'age', 'othinstplans',
        'housing', 'nbcred', 'job', 'nbliable', 'tel', 'foreign', 'badcredit'
    ]

    target = data['badcredit'] - 1
    
    data.drop('badcredit', axis=1, inplace=True)
    
    categorical = data.columns.difference(numerical + ['tel', 'foreign'])

    # Transform binary features
    data['tel'] = (data['tel'] == 'A191').astype(int)
    data['foreign'] = (data['foreign'] == 'A201').astype(int)

    if labeling:
        encoder = LabelEncoder()
        for c in categorical:
            data[c] = encoder.fit_transform(data[c])

    # One hot encoding of other categorical data
    if onehot:
        data = _binarize(data, categorical)

    data_train, target_train, data_tests, target_tests, sensitive = _gen_data(
        data, target, train_size, sensitive, **kwargs
    )

    del data

    if standardize:
        scaler = StandardScaler()
        scaler.fit(data_train[numerical])
        data_train[numerical] = scaler.transform(data_train[numerical])
        data_tests[numerical] = scaler.transform(data_tests[numerical])

    return data_train, target_train, data_tests, target_tests, sensitive


def load_arrhythmia_data(clean_missing=True, standardize=True, train_size=.7,
                         **kwargs):
    """ Loads Arrhythmia dataset from UCI Machine Learning website

        clean_missing: bool
            Remove missing data

        standardize: bool
            Transform data with mean > 1 to have 0 mean and 1 std.

        train_size: float
            Percentage of train data for train/test split.

        **kwargs: dict
            Any parameters to pass to the train test split function
            train_test_split

    Returns
    -------

        tuple of (X_train, X_test, y_train, y_test, sensitive)

    See Also
    --------

        sklearn.model_selection.train_test_split

    """

    data = _get_data(URL + 'arrhythmia/', 'arrhythmia.data', header=None)
    data.columns = (
        ['age', 'gender', 'heigh', 'weight', 'qrs_duration', 'pr_interval',
         'qt_interval', 't_interval', 'p_interval', 'qrs', 't', 'p', 'qrst',
         'j', 'heart_rate', 'avg_width_qwave', 'avg_width_rwave',
         'avg_width_swave', 'avg_width_rpwave', 'avg_width_spwave',
         'nb_deflections', 'ragged_rwave', 'diphasic_rwave', 'ragged_pwave',
         'dipahsic_pwave', 'ragged_twave', 'diphasic_twave']
        + [f'ch_dii{i}' for i in range(28, 40)]
        + [f'ch_diii{i}' for i in range(40, 52)]
        + [f'ch_avr{i}' for i in range(52, 64)]
        + [f'ch_avl{i}' for i in range(64, 76)]
        + [f'ch_avf{i}' for i in range(76, 88)]
        + [f'ch_v1{i}' for i in range(88, 100)]
        + [f'ch_v2{i}' for i in range(100, 112)]
        + [f'ch_v3{i}' for i in range(112, 124)]
        + [f'ch_v4{i}' for i in range(124, 136)]
        + [f'ch_v5{i}' for i in range(136, 148)]
        + [f'ch_v6{i}' for i in range(148, 160)]
        + ['amp_jwave', 'amp_rwave', 'amp_qwave', 'amp_swave',
           'amp_rpwave', 'amp_spwave', 'amp_pwave', 'amp_twave', 'qrsa',
           'qrsta']
        + [f'ch_dii{i}' for i in range(170, 180)]
        + [f'ch_diii{i}' for i in range(180, 190)]
        + [f'ch_avr{i}' for i in range(190, 200)]
        + [f'ch_avl{i}' for i in range(200, 210)]
        + [f'ch_avf{i}' for i in range(210, 220)]
        + [f'ch_v1{i}' for i in range(220, 230)]
        + [f'ch_v2{i}' for i in range(230, 240)]
        + [f'ch_v3{i}' for i in range(240, 250)]
        + [f'ch_v4{i}' for i in range(250, 260)]
        + [f'ch_v5{i}' for i in range(260, 270)]
        + [f'ch_v6{i}' for i in range(270, 280)] + ['arrhythmia'])

    sensitive = 'gender'
    if clean_missing:
        data.replace('?', np.nan, inplace=True)
        for c in data.columns:
            data[c] = data[c].fillna(data[c].mode()[0])

    data = data.astype(float)

    data[sensitive] = data[sensitive].astype(int)

    target = data['arrhythmia'].astype(int)
    target[target > 1] = 0

    # Drop all columns where 98% of the data are 0s
    mask = (~data.eq(0)).sum() < 10
    to_drop = data.loc[:, mask].columns.tolist()

    to_drop.append('arrhythmia')
    data.drop(to_drop, inplace=True, axis=1)

    data_train, target_train, data_tests, target_tests, sensitive = _gen_data(
        data, target, train_size, sensitive, **kwargs
    )
    
    del data

    if standardize:
        scaler = StandardScaler()
        scaler.fit(data_train)
        data_train[:] = scaler.transform(data_train)
        data_tests[:] = scaler.transform(data_tests)

    return data_train, target_train, data_tests, target_tests, sensitive


def _binarize(data, categorical) -> pd.DataFrame:
    cat2hot = make_column_transformer(
        (OneHotEncoder(sparse_output=False), categorical),
        remainder='passthrough'
    )
    cat2hot.set_output(transform='pandas')
    cat2hot.verbose_feature_names_out = False
    return cat2hot.fit_transform(data).astype(int)


def _get_data(url, filename, **kwargs) -> pd.DataFrame:
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col=0)
    df = pd.read_csv(url+filename, **kwargs)
    df.to_csv(filename)
    return df


def sample_distinct_cases(data_train, target_train, sensitive, indices=None):
    """ Make sure to have at least one case for each
    sensitive value and ground truth value
    """
    indices = indices or data_train.index
    
    x00 = np.random.choice(np.where(
        (target_train.loc[indices] == 0) &
        (data_train.loc[indices, sensitive] == 0))[0], 1)[0]

    x01 = np.random.choice(np.where(
        (target_train.loc[indices] == 0) &
        (data_train.loc[indices, sensitive] == 1))[0], 1)[0]

    x10 = np.random.choice(np.where(
        (target_train.loc[indices] == 1) &
        (data_train.loc[indices, sensitive] == 0))[0], 1)[0]

    x11 = np.random.choice(np.where(
        (target_train.loc[indices] == 1) &
        (data_train.loc[indices, sensitive] == 1))[0], 1)[0]
    
    return [x00, x01, x10, x11]


def _gen_data(data, target, train_size, sensitive, **kwargs):
    # We add 4 samples from the cross product {0, 1} x {0, 1} to
    # make sure that we have in the test set at least one sample
    # with ground truth in {0, 1} and sensitive feature with values
    # in {0, 1}
    ix = sample_distinct_cases(data, target, sensitive)
    xrows = data.iloc[ix]
    yrows = target.iloc[ix]
    
    data.drop(xrows.index, inplace=True)
    target.drop(yrows.index, inplace=True)

    data_train, data_tests, target_train, target_tests = train_test_split(
        data,
        target,
        train_size=train_size,
        **kwargs
    )

    data_tests = pd.concat([data_tests, xrows])
    target_tests = pd.concat([target_tests, yrows])
    return data_train, target_train, data_tests, target_tests, sensitive


def subsample_training(x, y, s, p=.5):
    si = np.random.choice(x.index, int(p * len(y)))
    return x.loc[si], y.loc[si], s.loc[si]


def folds_generator(x, y, s, n_splits=10, shuffle=False):
    
    folds = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    cases = set(sample_distinct_cases(x, y, s))
    
    for train, test in folds.split(x, y):
        to_test = set(train).intersection(cases)
        
        for elt in to_test:
            train = np.delete(train, np.where(train == elt))
            test = np.append(test, elt)

        yield train, test
