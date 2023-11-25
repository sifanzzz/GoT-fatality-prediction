fill_nan_cols = {'title':'No Title',
                'house':'No House',
                'dateOfBirth':'dateOfBirth',
                'culture':'Unknown'}

drop_cols = ['S.No',
            'plod',
            'name',
            'heir',
            'isAliveMother',
            'isAliveFather',
            'isAliveHeir',
            'isAliveSpouse',
            'father',
            'mother',
            'spouse',
            'age',
            'DateoFdeath']

less_frequent_cols = ['title','culture','house']





def preprocess_inputs(df,fill_nan_cols,drop_cols,less_frequent_cols):

  """
    Preprocesses a Pandas DataFrame by performing several data handling operations.

    Args:
    - df (pandas.DataFrame): The input DataFrame to be preprocessed.
    - fill_nan_cols (dict): A dictionary where keys represent column names and values
                            represent the values used to fill NaN values in those columns.
                            If 'dateOfBirth' is present, it uses the median for filling NaNs.
    - drop_cols (list): A list of column names to be dropped from the DataFrame.
    - less_frequent_cols (list): A list of column names where less frequent instances
                                 will be grouped together as 'Other'.

    Returns:
    - pandas.DataFrame: The preprocessed DataFrame after handling missing values,
                        dropping specified columns, and grouping less frequent instances.
    """

  df = df.copy()

  # Drop columns

  df = df.drop(drop_cols,axis=1)

  # Fill NaN values

  for key,val in fill_nan_cols.items():
      if key == 'dateOfBirth':
          df[key] = df[key].fillna(df[key].median())
      else:
          df[key] = df[key].fillna(val)

# Group less frequent instances

  keep_instances_dict = {}

  for col in less_frequent_cols:
    keep_instances = df[col].value_counts(normalize=True).reset_index().head(11).index.tolist()

    keep_instances_dict[col] = keep_instances

    df[col] = df[col].apply(lambda x: x if x in keep_instances_dict[col] else 'Other')

  return df
