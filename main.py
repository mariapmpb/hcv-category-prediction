import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import os


def exploratory_data_analysis(file_path, biomarkers):
    os.makedirs("figures", exist_ok=True)

    # opening the file
    df = pd.read_csv(file_path, sep=',',
                     names=['id', 'category', 'age', 'sex', 'alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea',
                            'ggt', 'prot'])

    # droping the first row
    df.drop(df.index[0], inplace=True)

    int_columns = ['age']
    df[int_columns] = df[int_columns].astype(int)
    float_columns = ['alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']
    df[float_columns] = df[float_columns].astype(float)

    print(
        '\n--------------------------------------------- EXPLORATORY DATA ANALYSIS ---------------------------------------------\n')
    print(f'# Observations and features:\n{df.shape}')
    print(f'\n# Top 5 observations:\n{df.head()}')
    print(f'\n# Last 5 observations:\n{df.tail()}')
    print(f'\n# Info about the dataframe:')
    df.info()
    print(f'\n# Number of unique elements:\n{df.nunique()}')
    print(f'\n# Missing values: \n {df.isnull().sum()}')
    print(f'\n# Duplicated values: {df.duplicated().sum()}')
    print(f'\n# Statistics:\n{df.describe(include="all").T}')

    for bm in biomarkers:
        print(f'\n---------------------------------------------')
        print(f'\n# Metrics {bm.upper()} before cleaning outliers: \n {df[bm].describe()}')

        plt.figure(figsize=(6, 4))
        plt.boxplot(df[bm].dropna())
        plt.title(f'Boxplot de {bm.upper()}')
        plt.ylabel(bm.upper())
        plt.tight_layout()
        plt.savefig(f'figures/boxplot_{bm}.png', dpi=300, bbox_inches='tight')
        plt.close()

    numeric_col = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_col = numeric_col.drop('id', errors='ignore')

    for c in numeric_col:
        # histogram for every numeric column
        plt.figure(figsize=(6, 4))
        df[c].dropna().hist(bins=30)
        plt.title(f'Distribuição de {c.upper()}')
        plt.xlabel(c.upper())
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.savefig(f'figures/hist_{c}.png', dpi=300, bbox_inches='tight')
        plt.close()

    corrrelation = df.drop(columns=['id']).corr(numeric_only=True)

    # heatmap to show the correlation between numeric variables
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrrelation, annot=True, fmt=".2f")
    plt.title('Correlação entre variáveis numéricas')
    plt.tight_layout()
    plt.savefig('figures/heatmap_correlacao.png', dpi=300, bbox_inches='tight')
    plt.close()

    # boxplot to show the distribution of category within age
    plt.figure(figsize=(7, 4))
    sns.boxplot(x='category', y='age', data=df)
    plt.title('Distribuição da idade por categoria clínica')
    plt.xlabel('Categoria')
    plt.ylabel('Idade')
    plt.tight_layout()
    plt.savefig('figures/boxplot_idade_por_categoria.png', dpi=300, bbox_inches='tight')
    plt.close()

    categorical_col = df.select_dtypes(include=['object', 'category']).columns
    categorical_col = categorical_col.drop('id', errors='ignore')

    for c in categorical_col:
        print(f'\n---------------------------------------------')
        print(f'\nSamples per value of {c}\n{df[c].value_counts()}')
        value_counts = df[c].value_counts()

        # barplot for every categorical column
        plt.figure(figsize=(7, 4))
        plt.bar(value_counts.index, value_counts.values)
        plt.title(f'Distribuição de {c}')
        plt.xlabel(c)
        plt.ylabel('Number of samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'figures/barplot_{c}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(
        '\n--------------------------------------------- EXPLORATORY DATA ANALYSIS ---------------------------------------------\n')

    return df


def encoding(df):
    # one-hot enconding for sex column
    df = pd.get_dummies(df, columns=['sex'], prefix='sex', drop_first=True, dtype=int)

    # encoding for category column
    encoder = LabelEncoder()
    df['category'] = encoder.fit_transform(df['category'])

    return df


def missing_values(df):
    # filling missing values
    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
        max_iter=10,
        random_state=42
    )

    array = imputer.fit_transform(df)
    df = pd.DataFrame(array, columns=df.columns)

    print(f'\n# Missing values after imputation: \n {df.isnull().sum()}')

    return df


def cleaning_outliers(df, biomarkers):
    # correcting outliers
    for bm in biomarkers:
        q1 = df[bm].quantile(0.25)
        q3 = df[bm].quantile(0.75)
        iqr = q3 - q1

        infw = q1 - 1.5 * iqr
        supw = q3 + 1.5 * iqr

        median = df[bm].median()

        df[bm] = np.where(
            (df[bm] < infw) | (df[bm] > supw),
            median,
            df[bm]
        )

        print(f'\n# Metrics {bm.upper()} after cleaning outliers: \n {df[bm].describe()}')

    return df


def normalizing(df, numeric_col):
    scaler = MinMaxScaler()
    numeric_col.append('age')

    df[numeric_col] = scaler.fit_transform(df[numeric_col])

    return df


def pre_processing(df, biomarkers):
    print(
        '\n--------------------------------------------- DATA PRE-PROCESSING ---------------------------------------------')

    # removing id column before preprocessing
    df = df.drop(columns=['id'])

    print('\n> Enconding categorical variables...')
    df = encoding(df)

    print('\n> Filling missing values...')
    df = missing_values(df)

    print('\n> Correcting outliers...')
    df = cleaning_outliers(df, biomarkers)

    print('\n> Normalizing data...')
    df = normalizing(df, biomarkers)

    print(f'\n> Statistics:\n{df.describe(include="all").T}')

    int_columns = ['category', 'sex_m']
    df[int_columns] = df[int_columns].astype(int)

    print(df)
    print(
        '\n--------------------------------------------- DATA PRE-PROCESSING ---------------------------------------------')

    return df


def data_splitting(df):
    X = df.drop(columns=['category'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    return X_train, X_test, y_train, y_test


def logistic_regression_model(X_train, X_test, y_train, y_test):
    os.makedirs("figures", exist_ok=True)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_predictions = lr.predict(X_test)

    print(f'\n# Logistic regression metrics report:')
    print(classification_report(y_test, y_predictions))

    cm = confusion_matrix(y_test, y_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matriz de Confusão - Regressão Logística')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_logistic_regression.png', dpi=300, bbox_inches='tight')
    plt.close()


def random_forest_model(X_train, X_test, y_train, y_test):
    os.makedirs("figures", exist_ok=True)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_predictions = rf.predict(X_test)

    print(f'\nAccuracy: {accuracy_score(y_test, y_predictions)}')
    print(f'\n# Random forest metrics report:')
    print(classification_report(y_test, y_predictions))

    cm = confusion_matrix(y_test, y_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matriz de Confusão - Random Forest')
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_random_forest.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    file_path = 'hcvdata/hcvdata.csv'
    biomarkers = ['alb', 'alp', 'alt', 'ast', 'bil', 'che', 'chol', 'crea', 'ggt', 'prot']

    initial_dataframe = exploratory_data_analysis(file_path, biomarkers)

    cleaned_dataframe = pre_processing(initial_dataframe, biomarkers)
    X_train, X_test, y_train, y_test = data_splitting(cleaned_dataframe)
    logistic_regression_model(X_train, X_test, y_train, y_test)
    random_forest_model(X_train, X_test, y_train, y_test)
