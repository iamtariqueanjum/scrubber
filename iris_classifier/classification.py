# Import required libraries
import streamlit as st  # Web app framework for creating interactive data applications
import pandas as pd  # Data manipulation and analysis library
from sklearn.datasets import load_iris  # Load the classic iris dataset
from sklearn.ensemble import RandomForestClassifier  # Machine learning classifier


@st.cache_data  # Streamlit decorator to cache the function result for better performance
def load_data():
    """
    Load and prepare the iris dataset for classification.
    
    Returns:
        tuple: (DataFrame with features and target, target class names)
    """
    iris = load_iris()  # Load the iris dataset (150 samples, 4 features, 3 classes)
    df = pd.DataFrame(iris.data, columns=iris.feature_names)  # Create DataFrame with feature data
    df["species"] = iris.target  # Add target column with numeric class labels (0, 1, 2)
    return df, iris.target_names  # Return DataFrame and human-readable class names


# Set up the Streamlit web app
st.title("Classification")  # Display main title for the web application

# Load the dataset using the cached function
df, target_names = load_data()

# Initialize and train the machine learning model
model = RandomForestClassifier()  # Create a Random Forest classifier instance
model.fit(df.iloc[:,:-1], df['species'])  # Train the model using all features except the last column (species)
# df.iloc[:,:-1] selects all rows and all columns except the last one (features only)
# df['species'] provides the target labels for training

st.sidebar.title("Input Features")  # Add a sidebar title

# Create a slider for each feature to select the range of values
sepal_length = st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))


input_data = [[sepal_length, sepal_width, petal_length, petal_width]] 

# prediction  
prediction = model.predict(input_data)
print(prediction)
predicted_species = target_names[prediction[0]]

# Output the prediction
st.write("Prediction")
st.write(f"The predicted species is: {predicted_species}")
