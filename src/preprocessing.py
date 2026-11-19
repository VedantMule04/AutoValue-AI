
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "year",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

CATEGORICAL_FEATURES = [
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand"
]


def create_preprocessor(scale_numeric=False):

    # Numerical preprocessing
    numerical_steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        ))
    ]

    if scale_numeric:
        numerical_steps.append(
            ("scaler", StandardScaler())
        )

    numerical_pipeline = Pipeline(
        steps=numerical_steps
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            ))
        ]
    )

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    X_train = pd.read_csv(
        BASE_DIR / "data" / "processed" / "X_train.csv"
    )

    preprocessor = create_preprocessor(
        scale_numeric=False
    )

    transformed_data = preprocessor.fit_transform(X_train)

    print("=" * 65)
    print("       AUTOVALUE AI - PREPROCESSING TEST")
    print("=" * 65)

    print("\nOriginal training shape:", X_train.shape)
    print("Transformed training shape:", transformed_data.shape)

    print("\nMissing values before preprocessing:")
    print(X_train.isnull().sum().sum())

    print("\nMissing values after preprocessing:")
    print(pd.DataFrame(transformed_data).isnull().sum().sum())

    print("\nPreprocessing pipeline executed successfully.")

    print("\n" + "=" * 65)
