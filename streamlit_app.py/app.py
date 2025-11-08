# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

import os

@st.cache_data
def load_data():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory level to the project root
    project_root = os.path.dirname(base_dir)
    # Construct the full path to the data file
    data_path = os.path.join(project_root, 'data', 'processed', 'resumes_processed.csv')
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one directory level to the project root
    project_root = os.path.dirname(base_dir)
    # Construct the full path to the model file
    model_path = os.path.join(project_root, 'data', 'processed', 'resume_classifier_lr.joblib')
    data_path = os.path.join(project_root, 'data', 'processed', 'resumes_processed.csv')
    
    # Load the model
    model = joblib.load(model_path)
    
    # Load the training data to get the category mapping
    try:
        df = pd.read_csv(data_path)
        if 'category' in df.columns:
            categories = sorted(df['category'].dropna().unique())
            category_mapping = {i: cat for i, cat in enumerate(categories)}
        else:
            category_mapping = None
    except Exception as e:
        print(f"Could not load category mapping: {e}")
        category_mapping = None
    
    # Check if this is a dictionary (new format) or just the pipeline (old format)
    if isinstance(model, dict):
        # New format with model data dictionary
        pipeline = model.get('pipeline', model)  # Fallback to model itself if 'pipeline' key doesn't exist
        label_encoder = model.get('label_encoder')
        target_col = model.get('target_col', 'category')
    else:
        # Old format - just the pipeline
        pipeline = model
        label_encoder = None
        target_col = 'category'
    
    return pipeline, label_encoder, target_col, category_mapping

df = load_data()
st.title("Resume Analysis Dashboard")

# KPIs
col1, col2 = st.columns(2)
with col1:
    total_resumes = st.number_input("Total resumes", min_value=0, value=len(df), step=1)
with col2:
    avg_exp = st.number_input("Average years experience", 
                            min_value=0.0, 
                            value=round(df['years_experience'].dropna().mean(), 2) if not df['years_experience'].dropna().empty else 0.0,
                            step=0.1,
                            format="%.1f")

# Display the metrics
col1.metric("Total resumes", total_resumes)
col2.metric("Average years experience", f"{avg_exp:.1f}")

# Filter
min_years = st.slider("Min years experience", 0, 30, 0)
filtered = df[df['years_experience'].fillna(0) >= min_years]

# Show top skills
all_skills = pd.Series([skill for skills in filtered['skills_list'].dropna() for skill in eval(str(skills))]).value_counts().head(20)
st.bar_chart(all_skills)

# Run model on a sample resume
st.header("Predict on custom resume text")
text = st.text_area("Paste resume text here")
if st.button("Predict"):
    try:
        pipeline, label_encoder, target_col, category_mapping = load_model()
        
        # Make prediction
        pred = pipeline.predict([text])[0]
        
        # Get the predicted label as a name
        pred_label = str(pred)  # Default to string representation
        
        # First, try to get the category name from the training data
        try:
            # Load the training data to get category names
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(base_dir)
            data_path = os.path.join(project_root, 'data', 'processed', 'resumes_processed.csv')
            df = pd.read_csv(data_path)
            
            if 'category' in df.columns:
                # Get unique categories and sort them
                categories = sorted(df['category'].dropna().unique())
                # Map prediction to category name
                if isinstance(pred, (int, float)) and 0 <= int(pred) < len(categories):
                    pred_label = categories[int(pred)]
        except Exception as e:
            print(f"Error getting category names: {e}")
        
        # If we still don't have a name, try other methods
        if pred_label == str(pred):
            if hasattr(pipeline, 'classes_') and isinstance(pred, (int, float)):
                try:
                    pred_label = pipeline.classes_[int(pred)]
                except (IndexError, TypeError):
                    pass
            elif hasattr(pipeline, 'named_steps') and 'clf' in pipeline.named_steps:
                clf = pipeline.named_steps['clf']
                if hasattr(clf, 'classes_') and isinstance(pred, (int, float)):
                    try:
                        pred_label = clf.classes_[int(pred)]
                    except (IndexError, TypeError):
                        pass
        
        st.write(f"**Predicted {target_col}:** {pred_label}")
        
        # Show prediction probabilities if available
        try:
            if hasattr(pipeline, 'predict_proba'):
                probs = pipeline.predict_proba([text])[0]
                
                # Try to get category names from the training data
                try:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(base_dir)
                    data_path = os.path.join(project_root, 'data', 'processed', 'resumes_processed.csv')
                    df = pd.read_csv(data_path)
                    
                    if 'category' in df.columns:
                        # Get unique categories and sort them
                        categories = sorted(df['category'].dropna().unique())
                        # Map indices to category names
                        if len(categories) == len(probs):
                            classes = categories
                        else:
                            classes = [str(i) for i in range(len(probs))]
                    else:
                        classes = [str(i) for i in range(len(probs))]
                except Exception as e:
                    print(f"Error getting category names: {e}")
                    classes = [str(i) for i in range(len(probs))]
                
                # Get top 3 predictions with their probabilities
                top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
                
                st.write("**Top predictions:**")
                for label, prob in top3:
                    st.write(f"- {label}: {prob*100:.1f}%")
                    
        except Exception as e:
            st.write("*Could not display prediction probabilities*")
            st.write(f"*Raw prediction value: {pred}*")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please try again or check the model file.")
        import traceback
        st.text(traceback.format_exc())
