import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Bin scores for reduced complexity
def bin_scores(df_range, bin_width=5):
    df_range = df_range.copy()
    df_range["fico_bin"] = (df_range["fico_score"] // bin_width) * bin_width
    grouped = df_range.groupby("fico_bin").agg(
        total=("default", "count"),
        defaults=("default", "sum")
    ).reset_index()
    return grouped

# Log-likelihood computation
def log_likelihood(n, k):
    if k == 0 or k == n:
        return 0  # Avoid log(0) — perfect bucket
    p = k / n
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Dynamic programming for optimal bucketing
def find_optimal_boundaries(data, num_buckets):
    N = len(data)
    dp = [[-np.inf] * (N + 1) for _ in range(num_buckets + 1)]
    path = [[-1] * (N + 1) for _ in range(num_buckets + 1)]
    dp[0][0] = 0

    for b in range(1, num_buckets + 1):
        for j in range(1, N + 1):
            for i in range(b - 1, j):
                total = data["total"].iloc[i:j].sum()
                defaults = data["defaults"].iloc[i:j].sum()
                ll = log_likelihood(total, defaults)
                if dp[b - 1][i] + ll > dp[b][j]:
                    dp[b][j] = dp[b - 1][i] + ll
                    path[b][j] = i

    # Backtrack to get boundaries
    boundaries = []
    curr = N
    for b in range(num_buckets, 0, -1):
        prev = path[b][curr]
        boundaries.append(data["fico_bin"].iloc[prev])
        curr = prev
    boundaries.reverse()
    return boundaries

# Apply to both ranges
fico_low = df[df["fico_score"] <= 600]
fico_high = df[df["fico_score"] > 600]
binned_low = bin_scores(fico_low)
binned_high = bin_scores(fico_high)

low_bounds = find_optimal_boundaries(binned_low, num_buckets=5)
high_bounds = find_optimal_boundaries(binned_high, num_buckets=5)

# Combine and sort all boundaries
all_bounds = low_bounds + high_bounds
all_bounds.sort()

# Rating assignment
def assign_rating(score, boundaries):
    for i, bound in enumerate(boundaries):
        if score <= bound:
            return i
    return len(boundaries)

# Add rating column
df["fico_rating"] = df["fico_score"].apply(lambda x: assign_rating(x, all_bounds))

# Export with ratings
df.to_csv("Loan_Data_With_FICO_Ratings.csv", index=False)

print("FICO bucketing completed. Output saved to 'Loan_Data_With_FICO_Ratings.csv'.")
