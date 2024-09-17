import pandas as pd


# Function to expand DataFrame based on new intervals
def expand_df(df, new_intervals):
    rows = []
    for _, row in df.iterrows():
        # Find matching new intervals
        matches = new_intervals[(new_intervals['t_s'] >= row['t_s']) & (new_intervals['t_e'] <= row['t_e'])]
        for _, match in matches.iterrows():
            # Copy the row and adjust the start and end times
            new_row = row.copy()
            new_row['t_s'] = match['t_s']
            new_row['t_e'] = match['t_e']
            rows.append(new_row)
    return pd.DataFrame(rows)


def unify_intervals(df1, df2):
    """
    Unifies the time intervals of two dataframes, duplicating rows and values as necessary to align intervals.

    Parameters:
    - df1 (pd.DataFrame): First DataFrame with columns 't_s', 't_e', and others.
    - df2 (pd.DataFrame): Second DataFrame with columns 't_s', 't_e', and others.

    Returns:
    - pd.DataFrame: Expanded version of df1 with unified intervals.
    - pd.DataFrame: Expanded version of df2 with unified intervals.
    """

    # Determine all unique start and end points
    all_times = sorted(set(df1['t_s'].tolist() + df1['t_e'].tolist() +
                           df2['t_s'].tolist() + df2['t_e'].tolist()))

    # Construct the new intervals
    new_intervals = pd.DataFrame({
        't_s': all_times[:-1],
        't_e': all_times[1:]
    })

    # Expand both dataframes
    df1_expanded = expand_df(df1, new_intervals)
    df2_expanded = expand_df(df2, new_intervals)

    return df1_expanded, df2_expanded


if __name__ == "__main__":
    dataframe1 = pd.DataFrame({
        't_s': [1, 3, 5],
        't_e': [3, 5, 10],
        'value': [100, 150, 200]
    })

    dataframe2 = pd.DataFrame({
        't_s': [1, 5, 6, 10],
        't_e': [5, 6, 9, 12],
        'value': [200, 250, 300, 350]
    })

    df1_expanded, df2_expanded = unify_intervals(dataframe1, dataframe2)
    print("Expanded df1:")
    print(df1_expanded)
    print("\nExpanded df2:")
    print(df2_expanded)
