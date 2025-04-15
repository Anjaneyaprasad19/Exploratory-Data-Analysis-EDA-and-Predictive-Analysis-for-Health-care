# main_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import warnings
import io
import os
from PIL import Image, ImageDraw, ImageFont
from streamlit_option_menu import option_menu  # Importing streamlit_option_menu
import streamlit.components.v1 as components  # For embedding HTML

warnings.filterwarnings('ignore')

# ----------------------------
# Configuration and Setup
# ----------------------------

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_PAGES_DIR = os.path.join(BASE_DIR, 'analysis_pages')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Dummy login credentials
USER_CREDENTIALS = {"username": "user", "password": "password"}

# ----------------------------
# Helper Functions
# ----------------------------

# CSS Styling (Removed Username and Password Input Styling)
def apply_css():
    st.markdown("""
        <style>
            /* Overall App Background matching the navigation bar */
            .stApp {
                background-color: #f5f5f5; /* Light gray */
            }
            /* Main Content Styling */
            .main-content {
                max-width: 1200px;
                margin: auto;
                padding: 20px;
                background-color: #2c3e50; /* Slightly lighter dark blue-gray */
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                color: #ecf0f1; /* Light gray text */
            }
            /* Titles and headers */
            .title {
                font-size: 2.5em;
                color: #007BFF; /* Blue heading font */
                text-align: center;
                margin-top: 20px;
                text-shadow: 1px 1px #34495e; /* Subtle text shadow */
                animation: fadeInDown 2s;
            }
            /* Logout button */
            .logout-button {
                position: absolute;
                bottom: 20px;
                width: 100%;
            }
            /* Animations */
            @keyframes fadeInDown {
                0% { opacity: 0; transform: translateY(-50px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            /* Card Styling */
            .card {
                border: 1px solid #7f8c8d;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #34495e; /* Darker card background */
                transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer;
                height: 200px;
                color: #ecf0f1; /* Light text */
            }
            .card:hover {
                transform: scale(1.05);
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            /* Footer Styling */
            .footer {
                text-align: center;
                padding: 10px;
                font-size: 0.9em;
                color: #7f8c8d;
            }
            /* Streamlit Sidebar Styling Override */
            [data-testid="stSidebar"] {
                background-color: #1a2634; /* Match nav bar */
                color: #ecf0f1; /* Light text */
            }
            /* Streamlit Option Menu Customization */
            .css-1d391kg {
                background-color: #1a2634;
            }
            .css-1v3fvcr {
                color: #ecf0f1;
            }
            /* Ensure all markdown text is light */
            .markdown-text-container {
                color: #ecf0f1;
            }
            /* Fullscreen Button Styling */
            .fullscreen-button {
                background-color: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            .fullscreen-button:hover {
                background-color: #2c3e50;
            }
        </style>
    """, unsafe_allow_html=True)

# Load datasets with caching for performance optimization
@st.cache_data
def load_project_data():
    st.info("📥Health Care Analysis...")
    time.sleep(1)  # Simulate loading time
    try:
        data_path = os.path.join(DATA_DIR, 'Project_healthcare_dataset.csv')
        return pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("Project_healthcare_dataset.csv not found in the data directory.")
        return pd.DataFrame()

@st.cache_data
def load_cleaned_data():
    st.info("📥 Loading Cleaned Dataset for Algorithm Predictions...")
    time.sleep(1)  # Simulate loading time
    try:
        data_path = os.path.join(DATA_DIR, 'Cleaned_healthcare_dataset.csv')
        return pd.read_csv(data_path)
    except FileNotFoundError:
        st.error("Cleaned_healthcare_dataset.csv not found in the data directory.")
        return pd.DataFrame()

# Cache model loading to improve performance
@st.cache_resource
def load_model(model_filename):
    model_path = os.path.join(MODELS_DIR, model_filename)
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"{model_filename} not found in the models directory.")
        return None

@st.cache_resource
def load_scaler(scaler_filename):
    scaler_path = os.path.join(MODELS_DIR, scaler_filename)
    try:
        return joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"{scaler_filename} not found in the models directory.")
        return None

@st.cache_resource
def load_encoders(encoders_filename):
    encoders_path = os.path.join(MODELS_DIR, encoders_filename)
    try:
        return joblib.load(encoders_path)
    except FileNotFoundError:
        st.error(f"{encoders_filename} not found in the models directory.")
        return None

# Cache HTML content to speed up rendering
@st.cache_data
def load_html_content(page_name):
    html_file = os.path.join(ANALYSIS_PAGES_DIR, f"{page_name}.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        st.error(f"❌ HTML file for '{page_name}' not found in analysis_pages directory.")
        return ""

# Login functionality
def login(username, password):
    if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
        st.session_state['logged_in'] = True
        st.success("🔓 Login successful!")
    else:
        st.error("❌ Invalid username or password")

# Logout functionality
def logout():
    st.session_state['logged_in'] = False

# Prediction function
def predict_test_result(model, scaler, encoders, new_data, features):
    # Validate and encode categorical data
    for feature in ['Gender', 'Blood Type', 'Medical Condition', 'Medication']:
        encoder = encoders.get(feature)
        if encoder is None:
            raise ValueError(f"Encoder for '{feature}' not found.")
        if new_data[feature] not in encoder.classes_:
            raise ValueError(f"Invalid value '{new_data[feature]}' for {feature}. Valid options are {list(encoder.classes_)}.")
        new_data[feature] = encoder.transform([new_data[feature]])[0]

    # Prepare data for prediction
    new_data_df = pd.DataFrame([new_data])[features]  # Ensure correct order of features
    new_data_scaled = scaler.transform(new_data_df)    # Scale the data

    # Predict and decode the result
    prediction = model.predict(new_data_scaled)
    result = encoders['Test Results'].inverse_transform(prediction)[0]
    return result

# Function to load and render HTML content with Fullscreen and Scroll
def render_html_page(page_name):
    html_content = load_html_content(page_name)
    if html_content:
        # Unique ID for the embedded content
        embedded_id = "embedded-content"

        # JavaScript for fullscreen toggle with escaped curly braces
        js_script = f"""
            <script>
                function toggleFullscreen() {{
                    var elem = document.getElementById('{embedded_id}');
                    if (!document.fullscreenElement) {{
                        elem.requestFullscreen().catch(err => {{
                            alert(`Error attempting to enable full-screen mode: ${{err.message}} ({{err.name}})`);
                        }});
                    }} else {{
                        document.exitFullscreen();
                    }}
                }}
            </script>
        """

        # Fullscreen Button HTML
        fullscreen_button = f"""
            <button class="fullscreen-button" onclick="toggleFullscreen()">Toggle Fullscreen</button>
        """

        # Container Div with embedded content
        container_html = f"""
            <div>
                {fullscreen_button}
                <div id="{embedded_id}" style=" border:1px solid #7f8c8d; 
                border-radius:10px; 
                padding:10px; 
                overflow: auto; 
                height: 1500px; 
                background-color:#C2C8CE; 
                color:#add8e6; >
                    {html_content}
                </div>
            </div>
            {js_script}
        """

        # Embed the HTML with the fullscreen functionality
        components.html(container_html, height=1600, scrolling=True)
    else:
        st.error(f"❌ Unable to load the HTML content for '{page_name}'.")

# ----------------------------
# Main Application Logic
# ----------------------------

def main():
    # Apply CSS styling
    apply_css()

    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        # Centered Login
        st.markdown('<h1 class="title">Welcome to Health Care Analysis 🏥</h1>', unsafe_allow_html=True)
        with st.form(key='login_form'):
            st.markdown("### Please log in to continue")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button(label='Login')
            if submit_button:
                login(username, password)
    else:
        # Custom Sidebar with streamlit-option-menu
        with st.sidebar:
            selected_page = option_menu(
                menu_title="Navigation",
                options=["Home", "Dataset Details", "Patient Details", "Analysis", "Algorithm Predictions", "Future Predictions", "Lifestyle Recommendations", "Contact Us"],
                icons=["house", "clipboard-data", "person-circle", "bar-chart", "gear", "activity", "heart", "envelope"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {
                        "padding": "5px", 
                        "background-color": "#1a2634",  # Dark teal/green background
                        "border-radius": "10px"
                    },
                    "icon": {
                       "color": "#ecf0f1",  # Light icons
                       "font-size": "18px"
                    },
                    "nav-link": {
                        "font-size": "16px",
                        "color": "#ecf0f1",  # Light text
                        "text-align": "left",
                        "margin": "5px",
                        "--hover-color": "#34495e",  # Slightly lighter shade for hover
                        "padding": "10px",
                    },
                    "nav-link-selected": {
                        "background-color": "#34495e",  # Lighter shade for selected item
                        "color": "#ecf0f1",  # Keep text white when selected
                        "font-weight": "bold"
                    },
                    "menu-title": {
                        "color": "#ecf0f1",  # Light title text
                        "font-size": "20px",
                        "padding": "10px 5px"
                    }
                }
             
            )

            if st.button("🚪 Logout"):
                logout()
                try:
                    st.experimental_rerun()
                except AttributeError:
                    st.error("The function `st.experimental_rerun()` is not available in your Streamlit version.")
                except Exception as e:
                    st.error(f"An error occurred while rerunning the app: {e}")

        page = selected_page

        # Load project dataset
        project_data = load_project_data()

        # Main content container
        with st.container():
            # Apply a div with class 'main-content' to style the content area
            #st.markdown('<div class="main-content">', unsafe_allow_html=True)

            # Home Page
            if page == "Home":
                st.markdown('<h1 class="title">Healthcare Data Analysis Project 🩺</h1>', unsafe_allow_html=True)
                st.write("""
                In the modern healthcare landscape, predictive analytics has emerged as a powerful tool for improving patient outcomes and optimizing healthcare delivery. With the increasing availability of healthcare data, there are opportunities to leverage machine learning to predict medical test results accurately.
    
                This project aims to develop machine learning models that predict medical test outcomes based on patient data. By leveraging state-of-the-art machine learning techniques, the project provides early predictions, improving healthcare decision-making and patient care.
    
                The project involves data preprocessing, feature engineering, model development, and evaluation. Various models were tested, and the best-performing models were optimized for deployment in a healthcare environment.
    
                **Project Highlights:**
                - **Predictive Modeling:** Utilize advanced algorithms to predict patient outcomes.
                - **Data Visualization:** Gain insights through comprehensive data visualization.
                - **Personalized Recommendations:** Offer tailored lifestyle and healthcare recommendations.
    
                **Use the sidebar to navigate through different features of the application.**
                """)

            # Dataset Details
            elif page == "Dataset Details":
                st.markdown('<h1 class="title">Dataset Details 📋</h1>', unsafe_allow_html=True)
                if not project_data.empty:
                    st.write("""
                    Our dataset consists of synthetic healthcare records, including patient demographics, medical conditions, medications, and test results.
    
                    **Sample Data:**
                    """)
                    st.dataframe(project_data.sample(100))
                else:
                    st.write("Dataset not available.")

            # Patient Details
            elif page == "Patient Details":
                st.markdown('<h1 class="title">Patient Details 🧑‍⚕️</h1>', unsafe_allow_html=True)
                st.write("Search for patient details by entering the patient's name.")
                patient_name = st.text_input("Enter Patient Name (e.g., Patient_150):")
                if st.button("Search"):
                    if not project_data.empty:
                        if patient_name in project_data['Name'].values:
                            patient_details = project_data[project_data['Name'] == patient_name].T
                            st.write(patient_details)
                            st.success("✅ Patient details retrieved successfully.")
                        else:
                            st.error("❌ Patient not found")
                    else:
                        st.error("Dataset not loaded.")

            # Analysis Section with Embedded HTML Pages
            elif page == "Analysis":
                st.markdown('<h1 class="title">Data Analysis 📊</h1>', unsafe_allow_html=True)

                # Define the HTML page names (without .html extension)
                links = {
                    "Data Loading and Preprocessing": "Data Loading and Preprocessing",
                    "Exploratory Data Analysis (EDA)": "EDA",
                    "Feature Engineering": "Feature Engineering",
                    "Model Building": "Model Building and Evaluation"
                }

                # Container for the cards
                with st.container():
                    cols = st.columns(4)

                    # Function to handle card clicks
                    def handle_card_click(page_key):
                        st.session_state['analysis_page'] = page_key

                    # Card 1: Preparing and Preprocessing
                    with cols[0]:
                        if st.button("Data Loading and Preprocessing", key="prep"):
                            handle_card_click("Data Loading and Preprocessing")
                        st.markdown(
                          f"""
                          <div class="card">
                             <h6></h6>
                             <p>Data cleaning, transformation, and preparation steps.</p>
                          </div>
                         """,
                         unsafe_allow_html=True
                          )

                    # Card 2: Exploratory Data Analysis (EDA)
                    with cols[1]:
                        if st.button("Exploratory Data Analysis (EDA)", key="eda"):
                            handle_card_click("EDA")
                        st.markdown(
                            f"""
                            <div class="card">
                                <h6></h6>
                                <p>Analyzing data distributions and relationships.</p>
                            </div>
                        """,
                        unsafe_allow_html=True
                       )

                    # Card 3: Feature Engineering
                    with cols[2]:
                        if st.button("Feature Engineering", key="fe"):
                            handle_card_click("Feature Engineering")
                        st.markdown(
                            f"""
                            <div class="card">
                                <h6></h6>
                                <p>Creating and selecting meaningful features.</p>
                            </div>
                            """,
                        unsafe_allow_html=True
                        )

                    # Card 4: Model Building
                    with cols[3]:
                        if st.button("Model Building and Evaluation", key="mb"):
                            handle_card_click("Model Building and Evaluation")
                        st.markdown(
                         f"""
                         <div class="card">
                            <h6></h6>
                            <p>Developing and evaluating predictive models.</p>
                        </div>
                        """,
                       unsafe_allow_html=True
                        )

                # Display the selected analysis HTML page
                if 'analysis_page' in st.session_state:
                    selected_analysis_page = st.session_state['analysis_page']
                    render_html_page(selected_analysis_page)
                else:
                    st.info("Click on a card above to view detailed analysis.")

            # Algorithm Predictions with Cleaned Dataset and Loading Spinner
            elif page == "Algorithm Predictions":
                st.markdown('<h1 class="title">Algorithm Predictions ⚙️</h1>', unsafe_allow_html=True)
                st.write("""
                **Select an algorithm** to train on the cleaned dataset and evaluate its performance.
                """)

                # Load cleaned dataset specifically for Algorithm Predictions
                data = load_cleaned_data()

                if not data.empty:
                    # Separate features and target
                    date_columns = ['Date of Admission', 'Discharge Date']
                    X = data.drop(columns=['Test Results_0', 'Test Results_1', 'Test Results_2', 'Name'] + date_columns, errors='ignore')
                    y = data[['Test Results_0', 'Test Results_1', 'Test Results_2']].idxmax(axis=1)
                    y = y.map({'Test Results_0': 0, 'Test Results_1': 1, 'Test Results_2': 2})
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)

                    # Define models
                    algorithm = st.selectbox("Choose an algorithm:", ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "K-Nearest Neighbors","Decision Tree"])
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42),
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                        "SVM": SVC(kernel='linear', probability=True, random_state=42),
                        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                    }

                    if st.button("Run Algorithm"):
                        with st.spinner('🔄 Running algorithm...'):
                            model = models[algorithm]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            accuracy = accuracy_score(y_test, y_pred)
                            report = classification_report(y_test, y_pred, target_names=["Normal", "Abnormal", "Inconclusive"], output_dict=True)

                            st.write(f"📊 **{algorithm} Results**")
                            st.write("**Accuracy:**", round(accuracy * 100, 2), "%")
                            st.write("**Classification Report:**")
                            st.table(pd.DataFrame(report).transpose())
                else:
                    st.error("Cleaned dataset not available.")

            # Future Prediction Section
            elif page == "Future Predictions":
                st.markdown('<h1 class="title">Future Prediction 🧬</h1>', unsafe_allow_html=True)
                st.write("Enter patient details to predict test results and receive personalized recommendations.")

                if not project_data.empty:
                    # Input form for new patient details
                    age = st.number_input("Age", min_value=0, max_value=120, value=30)
                    gender = st.selectbox("Gender", options=project_data['Gender'].unique())

                    # Blood Group selection
                    blood_group = st.selectbox("Blood Group", options=project_data['Blood Type'].unique())

                    # Continue with other fields
                    medical_condition = st.selectbox("Medical Condition", options=project_data['Medical Condition'].unique())
                    medication = st.selectbox("Medication", options=project_data['Medication'].unique())

                    if st.button("Predict"):
                        with st.spinner('Predicting...'):
                            # Prepare new patient data
                            new_patient = {
                                'Age': age,
                                'Gender': gender,
                                'Blood Type': blood_group,
                                'Medical Condition': medical_condition,
                                'Medication': medication
                            }

                            try:
                                # Load the trained model, scaler, and encoders with caching
                                model = load_model('random_forest_model.pkl')  # Update with your actual model filename
                                scaler = load_scaler('scaler.pkl')
                                encoders = load_encoders('label_encoders.pkl')

                                if model is not None and scaler is not None and encoders is not None:
                                    # Define feature order to align with model input
                                    features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Medication']

                                    # Perform the prediction
                                    result = predict_test_result(model, scaler, encoders, new_patient, features)
                                    st.write("🔍 **Predicted Test Result**:", result)

                                    # Generate image of the result (Optional: Can be removed if not desired)
                                    def generate_result_image(result_text):
                                        img = Image.new('RGB', (600, 300), color='#34495e')  # Darker background
                                        d = ImageDraw.Draw(img)
                                        try:
                                            font = ImageFont.truetype("arial.ttf", 60)
                                        except:
                                            font = ImageFont.load_default()
                                        text_width, text_height = d.textsize(result_text, font=font)
                                        position = ((img.width - text_width)/2, (img.height - text_height)/2)
                                        d.text(position, result_text, fill='#ecf0f1', font=font)  # Light text
                                        buf = io.BytesIO()
                                        img.save(buf, format='PNG')
                                        buf.seek(0)
                                        return buf

                                    # Uncomment the following lines if you want to display the image
                                    img_buf = generate_result_image(result)
                                    st.image(img_buf, caption='Prediction Result')

                                    # Generate health recommendations based on prediction
                                    recommendations = []
                                    if result == "Abnormal":
                                        recommendations.append("⚠️ **Follow-Up Testing**: Further specialized testing is recommended.")
                                        recommendations.append("💡 **Lifestyle Adjustments**: Adopt a healthier lifestyle with balanced nutrition and regular exercise.")
                                        recommendations.append("👩‍⚕️ **Specialist Consultation**: Consult a specialist for detailed evaluation.")
                                        # Update recommended doctors based on medical condition
                                        if medical_condition == "Hypertension":
                                            recommended_doctors = ["Dr. Matthew - Cardiologist", "Dr. Patel - Nephrologist"]
                                        elif medical_condition == "Diabetes":
                                            recommended_doctors = ["Dr. Kelly - Endocrinologist", "Dr. Jones - Diabetologist"]
                                        elif medical_condition == "Cancer":
                                            recommended_doctors = ["Dr. Lee - Oncologist", "Dr. Smith - Oncologist"]
                                        elif medical_condition == "Asthma":
                                            recommended_doctors = ["Dr. Taylor - Pulmonologist", "Dr. Johnson - Allergist"]
                                        else:
                                            recommended_doctors = ["Dr. Lee - Specialist", "Dr. Smith - Specialist"]
                                        # Update insurance suggestions based on existing providers
                                        insurance_suggestions = list(project_data['Insurance Provider'].unique())
                                    elif result == "Normal":
                                        recommendations.append("✅ **Routine Maintenance**: Continue with regular health check-ups.")
                                        recommendations.append("🏃 **Healthy Lifestyle**: Maintain your current healthy habits.")
                                        recommended_doctors = ["Dr. Jones - General Practitioner", "Dr. Kelly - Family Medicine"]
                                        insurance_suggestions = list(project_data['Insurance Provider'].unique())
                                    elif result == "Inconclusive":
                                        recommendations.append("🔄 **Re-Test Recommended**: Additional tests may be necessary.")
                                        recommendations.append("👀 **Monitor Symptoms**: Keep an eye on any new or worsening symptoms.")
                                        recommended_doctors = ["Dr. Smith - Diagnostic Specialist", "Dr. Patel - Internal Medicine"]
                                        insurance_suggestions = list(project_data['Insurance Provider'].unique())

                                    # Display health recommendations
                                    st.markdown("### Health Recommendations")
                                    for rec in recommendations:
                                        st.markdown(f"- {rec}")

                                    # Display recommended doctors
                                    st.markdown("### Recommended Doctors")
                                    for doc in recommended_doctors:
                                        st.markdown(f"- {doc}")

                                    # Display suggested insurance plans
                                    st.markdown("### Suggested Insurance Providers")
                                    for plan in insurance_suggestions:
                                        st.markdown(f"- {plan}")

                            except FileNotFoundError as e:
                                st.error(f"Model files not found: {e}")
                            except ValueError as e:
                                st.error(f"Error: {e}")
                else:
                    st.error("Project dataset not available.")

            # Lifestyle Recommendations Page
            elif page == "Lifestyle Recommendations":
                st.markdown('<h1 class="title">Lifestyle Recommendations 🌿</h1>', unsafe_allow_html=True)
                st.write("Find lifestyle adjustments and wellness tips based on specific medical conditions or medications.")

                recommendation_type = st.selectbox("Select Recommendation Type", ["Condition-Based", "Medication-Based"])

                if recommendation_type == "Condition-Based":
                    selected_condition = st.selectbox("Select a Medical Condition", project_data['Medical Condition'].unique())
                    st.write(f"### Lifestyle Recommendations for {selected_condition}")

                    # Provide detailed recommendations based on the selected condition
                    if selected_condition == "Hypertension":
                        st.markdown("""
                        **Hypertension Management Tips:**
                        - **Dietary Approaches:**
                            - Reduce sodium intake to less than 2,300 mg per day.
                            - Incorporate potassium-rich foods like bananas, spinach, and avocados.
                            - Follow the DASH diet, emphasizing fruits, vegetables, whole grains, and low-fat dairy.
                        - **Physical Activity:**
                            - Engage in at least 150 minutes of moderate aerobic exercise per week.
                            - Include strength training exercises at least twice a week.
                        - **Lifestyle Changes:**
                            - Limit alcohol consumption.
                            - Quit smoking and avoid secondhand smoke.
                            - Manage stress through meditation, yoga, or deep-breathing exercises.
                        - **Regular Monitoring:**
                            - Check blood pressure regularly and keep a log for healthcare visits.
                        """)
                    elif selected_condition == "Diabetes":
                        st.markdown("""
                        **Diabetes Management Tips:**
                        - **Healthy Eating:**
                            - Focus on a balanced diet with controlled carbohydrate intake.
                            - Choose high-fiber foods like whole grains, legumes, fruits, and vegetables.
                            - Limit sugary beverages and processed foods.
                        - **Physical Activity:**
                            - Aim for at least 150 minutes of moderate aerobic activity weekly.
                            - Include resistance training to improve insulin sensitivity.
                        - **Blood Sugar Monitoring:**
                            - Check blood glucose levels as recommended.
                            - Be aware of symptoms of hypoglycemia and hyperglycemia.
                        - **Foot Care:**
                            - Inspect feet daily for cuts, blisters, or sores.
                            - Wear comfortable shoes and keep feet clean and dry.
                        - **Regular Check-ups:**
                            - Schedule routine appointments for A1C tests, eye exams, and kidney function tests.
                        """)
                    elif selected_condition == "Cancer":
                        st.markdown("""
                        **Cancer Management Tips:**
                        - **Nutrition:**
                            - Eat a balanced diet rich in fruits, vegetables, and whole grains.
                            - Limit red and processed meats.
                        - **Physical Activity:**
                            - Engage in regular physical activity as tolerated.
                            - Focus on exercises that improve strength and flexibility.
                        - **Emotional Support:**
                            - Seek support from counseling or support groups.
                            - Practice stress-reduction techniques like mindfulness and meditation.
                        - **Follow-up Care:**
                            - Keep regular appointments with your oncology team.
                            - Stay informed about your treatment plan and any side effects.
                        """)
                    elif selected_condition == "Asthma":
                        st.markdown("""
                        **Asthma Management Tips:**
                        - **Avoid Triggers:**
                            - Identify and avoid allergens like pollen, dust mites, and pet dander.
                            - Use air purifiers and maintain clean indoor air.
                        - **Medication Adherence:**
                            - Take prescribed inhalers and medications consistently.
                            - Understand the difference between rescue and maintenance inhalers.
                        - **Lifestyle Modifications:**
                            - Exercise regularly but be cautious of exercise-induced asthma.
                            - Practice breathing exercises to strengthen lung capacity.
                        - **Environmental Control:**
                            - Keep humidity levels low to prevent mold growth.
                            - Avoid exposure to smoke and strong odors.
                        - **Asthma Action Plan:**
                            - Work with your doctor to develop a personalized action plan.
                            - Monitor symptoms and know when to seek medical help.
                        """)
                    else:
                        st.write("Specific recommendations for this condition are not available at the moment.")

                elif recommendation_type == "Medication-Based":
                    selected_medication = st.selectbox("Select a Medication", project_data['Medication'].unique())
                    st.write(f"### Lifestyle Recommendations for {selected_medication}")

                    # Provide detailed recommendations based on the selected medication
                    if selected_medication == "Aspirin":
                        st.markdown("""
                        **Aspirin Usage Guidelines:**
                        - **Dosage:**
                            - Follow the prescribed dosage; do not exceed recommended amounts.
                        - **Food Interactions:**
                            - Take with food or a full glass of water to minimize stomach upset.
                        - **Alcohol Consumption:**
                            - Limit alcohol intake to reduce the risk of gastrointestinal bleeding.
                        - **Monitoring:**
                            - Be aware of signs of bleeding, such as unusual bruising or black stools.
                        """)
                    elif selected_medication == "Ibuprofen":
                        st.markdown("""
                        **Ibuprofen Usage Guidelines:**
                        - **Dosage:**
                            - Use the lowest effective dose for the shortest duration necessary.
                        - **Food Interactions:**
                            - Taking with food or milk can reduce stomach irritation.
                        - **Avoid Combining with NSAIDs:**
                            - Do not use with other NSAIDs to prevent increased risk of side effects.
                        - **Monitoring:**
                            - Watch for signs of gastrointestinal bleeding or kidney issues.
                        """)
                    elif selected_medication == "Paracetamol":
                        st.markdown("""
                        **Paracetamol Usage Guidelines:**
                        - **Dosage:**
                            - Do not exceed 4,000 mg in a 24-hour period.
                        - **Alcohol Consumption:**
                            - Avoid alcohol to reduce the risk of liver damage.
                        - **Monitoring:**
                            - Be cautious of other medications containing paracetamol to prevent overdose.
                        """)
                    elif selected_medication == "Penicillin":
                        st.markdown("""
                        **Penicillin Usage Guidelines:**
                        - **Administration:**
                            - Take as prescribed; complete the full course even if symptoms improve.
                        - **Allergy Awareness:**
                            - Inform your doctor if you have any known allergies to penicillin or other antibiotics.
                        - **Side Effects:**
                            - Report any signs of allergic reactions, such as rash, itching, or difficulty breathing.
                        """)
                    else:
                        st.write("Specific recommendations for this medication are not available at the moment.")

            # Contact Us Section
            elif page == "Contact Us":
                st.markdown('<h1 class="title">Contact Us 📞</h1>', unsafe_allow_html=True)
                st.write("""
                **We'd love to hear from you!**
    
                - **Address:** 2801 S University Ave, Little Rock, AR 72204
                - **Phone:** +1 (123) 456-7890
                - **Email:** healthcare@example.com
    
                *Feel free to reach out with any questions, feedback, or inquiries about our services.*
                """)

            # Footer
            elif page == "Contact Us":
                st.markdown('<div class="footer">', unsafe_allow_html=True)
                st.markdown("---")
                st.write("© 2024 Healthcare Data Analysis Platform")

                # Data Privacy Notice in the footer
                st.markdown("**Data Privacy Notice:** The data used is synthetic. For real data, ensure compliance with data protection regulations.")

                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()