# Prepare data for predictive modeling (example: predict role/category or label if available)
# Some datasets include a 'category' or 'tag' column; adapt accordingly.
target_col = None
for c in df.columns:
    if c in ['category','job_category','label','target','designation']:
        target_col = c
        break

# If dataset has no target, you can create a heuristic target (e.g., classify into 'data_science' vs 'non_data')
# Example: create binary label whether resume contains 'data' or 'machine learning'
if target_col is None:
    df['is_data_role'] = df['resume_clean'].apply(lambda x: 1 if ('data' in x or 'machine learning' in x or 'ml ' in x) else 0)
    target_col = 'is_data_role'

print("Using target:", target_col, "counts:\n", df[target_col].value_counts())
