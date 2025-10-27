# Save processed data and artifacts
processed_path = DATA_PATH / "processed"
processed_path.mkdir(exist_ok=True)
df.to_csv(processed_path / "resumes_processed.csv", index=False)
# Save model (optional)
import joblib
joblib.dump(pipeline, processed_path / "resume_classifier_lr.joblib")
