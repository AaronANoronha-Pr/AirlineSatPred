import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Function to load the model
def load_model(model_name):
    if model_name == "Naive Bayes":
        model = joblib.load('naive_bayes_model.pkl')
    elif model_name == "Random Forest":
        model = joblib.load('random_forest_model.pkl')
    else:
        raise ValueError("Invalid model name")
    return model

# Function to make prediction
def predict_satisfaction(model, user_input):
    # Columns present in the dataset
    columns = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
               'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
               'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 
               'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 
               'Checkin service', 'Inflight service', 'Cleanliness', 'Departure Delay in Minutes', 
               'Arrival Delay in Minutes']
    
    # Convert user input into DataFrame
    input_df = pd.DataFrame(user_input, index=[0])
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df)
    
    # Reorder columns to match the model's input order
    input_df = input_df.reindex(columns=columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]

def satisfaction_prediction():
    st.title("Satisfaction Prediction")
    
    selected_model = st.selectbox("Select Model", ["Naive Bayes", "Random Forest"])

    # Load the model
    model = load_model(selected_model)

    # Get user inputs
    user_input = {}

    # Define the number of inputs in each column
    num_inputs_per_column = 5

    # Define columns for the inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        user_input['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
        user_input['Customer Type'] = st.selectbox("Customer Type", ['Loyal Customer', 'disloyal Customer'])
        user_input['Age'] = st.number_input("Age", min_value=0, max_value=120, value=30)
        user_input['Type of Travel'] = st.selectbox("Type of Travel", ['Personal Travel', 'Business travel'])
        user_input['Class'] = st.selectbox("Class", ['Eco Plus', 'Business', 'Eco'])
        user_input['Flight Distance'] = st.number_input("Flight Distance", min_value=0, value=500)
        user_input['Inflight wifi service'] = st.slider("Inflight wifi service rating (1-5)", 1, 5, 3)
        user_input['Departure/Arrival time convenient'] = st.slider("Departure/Arrival time convenience rating (1-5)", 1, 5, 3)

    with col2:
        user_input['Ease of Online booking'] = st.slider("Ease of Online booking rating (1-5)", 1, 5, 3)
        user_input['Gate location'] = st.slider("Gate location rating (1-5)", 1, 5, 3)
        user_input['Food and drink'] = st.slider("Food and drink rating (1-5)", 1, 5, 3)
        user_input['Online boarding'] = st.slider("Online boarding rating (1-5)", 1, 5, 3)
        user_input['Seat comfort'] = st.slider("Seat comfort rating (1-5)", 1, 5, 3)
        user_input['Inflight entertainment'] = st.slider("Inflight entertainment rating (1-5)", 1, 5, 3)
        user_input['On-board service'] = st.slider("On-board service rating (1-5)", 1, 5, 3)

    with col3:
        user_input['Leg room service'] = st.slider("Leg room service rating (1-5)", 1, 5, 3)
        user_input['Baggage handling'] = st.slider("Baggage handling rating (1-5)", 1, 5, 3)
        user_input['Checkin service'] = st.slider("Checkin service rating (1-5)", 1, 5, 3)
        user_input['Inflight service'] = st.slider("Inflight service rating (1-5)", 1, 5, 3)
        user_input['Cleanliness'] = st.slider("Cleanliness rating (1-5)", 1, 5, 3)
        user_input['Departure Delay in Minutes'] = st.number_input("Departure Delay in Minutes", min_value=0, value=0)

        # For the last input, use a number_input for consistency
        user_input['Arrival Delay in Minutes'] = st.number_input("Arrival Delay in Minutes", min_value=0.0, value=0.0)

    # Predict satisfaction level
    if st.button("Predict Satisfaction"):
        prediction = predict_satisfaction(model, user_input)
        st.write("Predicted satisfaction level:", prediction)

def feature_analysis():
    st.title("Feature Analysis")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data.dropna(inplace=True)

        satisfaction_levels = data['satisfaction'].unique()
        selected_satisfaction = st.selectbox("Select Satisfaction Level", satisfaction_levels)

        display_data = data[data['satisfaction'] == selected_satisfaction]

        display_mean_values(display_data)
        display_plots(display_data)

def display_mean_values(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'id']
    numeric_mean_values = data[numeric_columns].mean()

    mean_table_data = pd.DataFrame({
        'Features': numeric_mean_values.index,
        'Influence on Output': numeric_mean_values.values
    })

    st.write("#### Mean Values:")
    st.table(mean_table_data)

def display_plots(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    data_type_of_travel = data.groupby('Type of Travel').size()
    data_type_of_travel.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, ax=axes[0])
    axes[0].set_title(f'Pie Chart for {data.iloc[0]["satisfaction"]} Customers (Type of Travel)')
    axes[0].set_ylabel('')
    plt.tight_layout()

    data_class = data.groupby('Class').size()
    data_class.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, ax=axes[1])
    axes[1].set_title(f'Pie Chart for {data.iloc[0]["satisfaction"]} Customers (Class)')
    axes[1].set_ylabel('')
    plt.tight_layout()

    st.pyplot(fig)

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Satisfaction Prediction", "Feature Analysis"])

    if page == "Satisfaction Prediction":
        satisfaction_prediction()
    elif page == "Feature Analysis":
        feature_analysis()
