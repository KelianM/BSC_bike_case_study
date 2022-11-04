import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import matplotlib.pyplot as plt


class BikeDataManager:
    def __init__(self, filename):
        self.bike_data = pd.read_csv(filename, usecols=range(13))
        # Seperate target and features
        self.y = self.bike_data[['Purchased Bike']].copy()
        self.X = self.bike_data.drop(['ID', 'Purchased Bike'], axis=1).copy()

    def show_summary(self):
        df = self.bike_data
        print("Dataset features:")
        print(df.columns)
        print("=" * 100)
        print("Example data:")
        print(df.head())
        print("=" * 100)
        print("Column summaries:")
        # Show unique values for each column
        for column in df:
            if column == 'ID':
                continue
            unique = df[column].unique()
            print(df[column].describe())
            if len(unique) < 10:
                print(f"Unique values:")
                print(df[column].unique())
            print("=" * 50)
        print("=" * 100)

    def standardise(self):
        num_cols = [col for col in self.X.columns if self.X[col].dtype != 'object']
        if len(num_cols) > 0:
            scaler = StandardScaler()
            self.X[num_cols] = scaler.fit_transform(self.X[num_cols])

    def handle_missing(self):
        # Replace NaNs
        cols_with_missing = [col for col in self.X.columns
                             if self.X[col].isnull().any()]
        if len(cols_with_missing) > 0:
            print(f"Imputing for missing values in cols: {cols_with_missing}")
            imputer = SimpleImputer(strategy='most_frequent')
            self.X[cols_with_missing] = imputer.fit_transform(self.X[cols_with_missing])
        # Relace -1's
        cols_with_missing = [col for col in self.X.columns
                             if self.X[col].dtype != 'object' and (self.X[col] < 0).any()]
        if len(cols_with_missing) > 0:
            print(f"Imputing for invalid values (-1) in cols: {cols_with_missing}")
            imputer = SimpleImputer(strategy='most_frequent', missing_values=-1)
            self.X[cols_with_missing] = imputer.fit_transform(self.X[cols_with_missing])

    def encode(self):
        # Apply one-hot encoder to Region column
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=int)
        OH_cols = pd.DataFrame(OH_encoder.fit_transform(self.X[['Region']]))
        # One-hot encoding removed index; put it back
        OH_cols.index = self.X.index
        # Add names for OH-encoded features
        OH_cols.columns = OH_encoder.get_feature_names_out(['Region'])
        # Replace Region column with OH encoded values
        X_no_OH = self.X.drop(['Region'], axis=1)
        self.X = pd.concat([X_no_OH, OH_cols], axis=1)
        # Encode ordinal columns with an order
        cols = ['Commute Distance', 'Occupation', 'Education']
        cats = [['0-1 Miles', '1-2 Miles', '2-5 Miles', '5-10 Miles', '10+ Miles'],
                ['Manual', 'Clerical', 'Skilled Manual', 'Professional', 'Management'],
                ['Partial High School', 'High School', 'Partial College', 'Bachelors', 'Graduate Degree']]
        ordinal_encoder = OrdinalEncoder(dtype=int, categories=cats)
        self.X[cols] = ordinal_encoder.fit_transform(self.X[cols])
        # Then encode the rest of the categorical features with default ordinal encoding
        ordinal_encoder = OrdinalEncoder(dtype=int)
        # Encode target
        self.y = ordinal_encoder.fit_transform(self.y.values.reshape(-1, 1)).ravel()
        # Get list of remaining categorical features
        s = (self.X.dtypes == 'object')
        object_cols = list(s[s].index)
        # Apply ordinal encoder to remaining categorical features
        self.X[object_cols] = ordinal_encoder.fit_transform(self.X[object_cols])
        self.X = self.X.astype(int)

    def mutual_info(self, plot=False, thresh=1e-3):
        # Calculate MI scores for all features and drop features that have an MI below the threshold
        discrete_features = self.X.dtypes == int
        mi_scores = mutual_info_classif(self.X, self.y, discrete_features=discrete_features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=self.X.columns)
        mi_scores = mi_scores.sort_values(ascending=True)

        if plot:
            plt.figure(dpi=100, figsize=(8, 5))
            width = np.arange(len(mi_scores))
            ticks = list(mi_scores.index)
            plt.barh(width, mi_scores)
            plt.yticks(width, ticks)
            plt.title("Mutual Information Scores")
            plt.show()

            # Second zoomed plot for lowest scores
            plt.figure(dpi=100, figsize=(5, 3))
            mi_scores = mi_scores[:3]
            width = np.arange(len(mi_scores))
            ticks = list(mi_scores.index)
            plt.barh(width, mi_scores)
            plt.yticks(width, ticks)
            plt.title("Mutual Information Scores (zoomed)")
            plt.show()

        # Remove columns with MI less than thresh
        low_mi_score_cols = []
        for col, mi_score in mi_scores.items():
            if mi_score < thresh:
                low_mi_score_cols.append(col)
        print(f"Dropping columns with low MI scores: {low_mi_score_cols}")
        self.X = self.X.drop(low_mi_score_cols, axis=1)

    def plot_learning_curves(
            self,
            estimator,
            title,
            axes=None,
            ylim=None,
            cv=None,
            scoring=None,
            train_sizes=np.linspace(0.1, 1.0, 5)
    ):
        """
        Generate 3 plots: the test and training learning curve, the training
        samples vs fit times curve, the fit times vs score curve.
        """
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Average Precison (%)")

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            estimator,
            self.X, self.y,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            train_sizes=train_sizes,
            return_times=True,
        )
        # Convert to %
        train_scores *= 1e2
        test_scores *= 1e2
        # Convert to ms
        fit_times *= 1e3
        # Get mean and std of metrics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g"
        )
        axes[0].plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        axes[0].plot(
            train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times (scalability)
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            train_sizes,
            fit_times_mean - fit_times_std,
            fit_times_mean + fit_times_std,
            alpha=0.1,
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("Training Time (ms)")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score (performance)
        fit_time_argsort = fit_times_mean.argsort()
        fit_time_sorted = fit_times_mean[fit_time_argsort]
        test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
        test_scores_std_sorted = test_scores_std[fit_time_argsort]
        axes[2].grid()
        axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
        axes[2].fill_between(
            fit_time_sorted,
            test_scores_mean_sorted - test_scores_std_sorted,
            test_scores_mean_sorted + test_scores_std_sorted,
            alpha=0.1
        )
        axes[2].set_xlabel("Training Time (ms)")
        axes[2].set_ylabel("Average Precision (%)")
        axes[2].set_title("Performance of the model")
        return plt