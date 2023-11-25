from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_pipeline():
    """
    Creates a preprocessing pipeline for transforming features of a dataset.

    Numeric Features:
        - "dateOfBirth": Scaled using StandardScaler.

    Categorical Features:
        - "title", "culture", "house": Encoded using OneHotEncoder. The encoder
          handles unknown categories by ignoring them, and binary categories are
          dropped if they are redundant.

    Passthrough Features:
        - "male", "book1", "book2", "book3", "book4", "book5", "isMarried",
          "isNoble", "numDeadRelations", "boolDeadRelations", "isPopular",
          "popularity": These features are passed through without any transformation.

    Returns:
        A sklearn ColumnTransformer object that applies the specified transformations
        to the appropriate columns.
    """
    numeric_features = ["dateOfBirth"]
    categorical_features = ["title", "culture", "house"]
    passthrough_features = [
        "male",
        "book1",
        "book2",
        "book3",
        "book4",
        "book5",
        "isMarried",
        "isNoble",
        "numDeadRelations",
        "boolDeadRelations",
        "isPopular",
        "popularity",
    ]

    # Create the column transformer
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, drop="if_binary"
    )
    # binary_transformer = OneHotEncoder(sparse_output=False, dtype = int)

    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        ("passthrough", passthrough_features),
        (categorical_transformer, categorical_features),
    )

    return preprocessor
