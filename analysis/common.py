import numpy as np
import pandas as pd

# metrics for similarity, dtw, smith-waterman
METRIC_COSINE = 'cosine'
METRIC_EUCLIDEAN = 'euclidean'


# vector extraction methods
def parse_vectors(df: pd.DataFrame, body_part, normalize: bool = False) -> np.ndarray:
    """
    parses the input dataframe to extract vector data for a specific body part,
    optionally normalizing the vectors. returns a structured numpy array.

    parameters:
    - df: pandas dataframe containing vector data
    - body_part: column name containing the vector data
    - normalize: boolean, if true, normalizes each vector

    returns:
    - numpy structured array containing vectors and timestamps
    """

    # check if the DataFrame is None or empty
    if df is None or df.empty:
        print("parse_vectors: DataFrame is either None or empty.")
        return None

    # check if the body_part column exists in the DataFrame
    if body_part not in df.columns:
        print(f"parse_vectors: Column '{body_part}' not found in the DataFrame.")
        return None

    # fix to handle both strings and numpy arrays
    df[body_part] = df[body_part].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
    )

    if normalize:
        df[body_part] = df[body_part].apply(normalize_vector)

    # structured data type for the numpy array
    dt = np.dtype([
        (body_part, np.float64, (3,)),  # assuming each vector has 3 components
        ('t_s', np.float64),
        ('t_e', np.float64)
    ])

    # prepare the data
    structured_array = np.zeros(df.shape[0], dtype=dt)
    structured_array[body_part] = np.stack(df[body_part].values)
    structured_array['t_s'] = df['t_s'].values
    structured_array['t_e'] = df['t_e'].values

    return structured_array


def normalize_vector(v: np.ndarray):
    """
    normalizes the input vector.

    parameters:
    - v: numpy array representing the vector to be normalized

    returns:
    - normalized vector (numpy array) if norm is not zero, else returns the original vector
    """
    # flatten the input vector if needed
    v = np.array(v).flatten()
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v
