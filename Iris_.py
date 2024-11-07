import streamlit as st 
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset and rename columns for easier access
data = load_iris(as_frame=True).frame
data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

# Display the dataset
st.write("### Iris Dataset")
st.write(data.head())

# Add data description and statistics
st.write("Data Description")
st.write(data.describe())

# Filter data based on sepal length
sepal_length = st.sidebar.slider("Filter by Sepal Length", float(data["sepal_length"].min()), float(data["sepal_length"].max()))
filtered_data = data[data["sepal_length"] >= sepal_length]
st.write("Filtered Data", filtered_data)

# Visualize data with Seaborn
st.subheader("Sepal Length vs. Sepal Width")
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x="sepal_length", y="sepal_width", hue="target", palette="viridis", ax=ax)
st.pyplot(fig)

# Model training and accuracy display
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write("Model Accuracy:", accuracy)

# Predict based on user inputs
st.write("### Make a Prediction")
sepal_length_input = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width_input = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length_input = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
petal_width_input = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

if st.button("Predict"):
    prediction = model.predict([[sepal_length_input, sepal_width_input, petal_length_input, petal_width_input]])
    st.write("Predicted Class:", prediction[0])
