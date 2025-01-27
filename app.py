import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import *
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import matplotlib.pyplot as plt
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn
import io
import base64
import os

##############################################################################
# 1. LOAD & CLEAN DATA
##############################################################################
data = pd.read_csv("pdsvul.csv")

# Replace non-alphanumeric chars with underscores in column names:
data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

# If 'S_No' is present but not needed, drop it to avoid KeyErrors:
data.drop(columns=['S_No'], errors='ignore', inplace=True)

# Convert 'Vulnerable' to numeric if needed ("yes"/"no" -> 1/0):
if 'Vulnerable' in data.columns:
    data['Vulnerable'] = data['Vulnerable'].replace({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})

##############################################################################
# 2. DEFINE X AND y
##############################################################################
X = data.drop(columns=['Vulnerable'], errors='ignore')
y = data['Vulnerable'].astype(int)

##############################################################################
# 3. ENCODE CATEGORICAL FEATURES & SANITIZE COLUMN NAMES
##############################################################################
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

encoder = OneHotEncoder(drop='first', sparse_output=False)
if len(cat_cols) > 0:
    X_encoded = encoder.fit_transform(X[cat_cols])
    original_feature_names = encoder.get_feature_names_out(cat_cols)

    # Replace forbidden chars (periods, braces) with underscores
    sanitized_feature_names = [
        f.replace('.', '_').replace('{', '_').replace('}', '_')
        for f in original_feature_names
    ]

    X_encoded_df = pd.DataFrame(X_encoded, columns=sanitized_feature_names)
else:
    X_encoded_df = pd.DataFrame()
    sanitized_feature_names = []

# Combine numeric + encoded
X_final = pd.concat(
    [X[numeric_cols].reset_index(drop=True),
     X_encoded_df.reset_index(drop=True)],
    axis=1
)

##############################################################################
# 4. SPLIT TRAIN/TEST
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

##############################################################################
# 5. TRAIN RANDOM FOREST
##############################################################################
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

##############################################################################
# 6. MODEL PERFORMANCE
##############################################################################
y_pred = rf_classifier.predict(X_test)
y_prob = rf_classifier.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

##############################################################################
# 7. CREATE EXPLAINER
##############################################################################
explainer = ClassifierExplainer(rf_classifier, X_test, y_test)

##############################################################################
# 8. PRECISION-RECALL PLOT
##############################################################################
def create_precision_recall_plot():
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(rec_vals, prec_vals, marker='.', label='RandomForest')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

##############################################################################
# 9. CUSTOM DASHBOARD
##############################################################################
class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, name=name)

        # Standard components:
        self.shap_summary = ShapSummaryComponent(explainer)
        self.feature_importance = ImportancesComponent(explainer)
        self.confusion_matrix = ConfusionMatrixComponent(explainer)
        self.feature_input = FeatureInputComponent(explainer)
        
        # Dependence Plot (pick a valid column in X_test, e.g. "Gender_Male")
        self.dependence = ShapDependenceComponent(explainer, col="Gender_Male")

        # Precision-Recall image
        self.pr_curve_img = create_precision_recall_plot()

    def layout(self):
        return dbc.Container(
            [
                dbc.Row(
                    [
                        # LEFT: Overview & Prediction
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H4("Model Overview, Inputs, and Prediction", className="text-primary")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5("Classification Problem", className="mt-3"),
                                            html.P(
                                                "This application predicts whether an individual is 'Vulnerable' based on "
                                                "demographic & socio-economic features such as Gender, Education, "
                                                "Marital Status, Primary Occupation, and Total Income.",
                                                className="card-text text-muted",
                                                style={'fontSize': '1.1rem'}
                                            ),

                                            html.H5("Random Forest Model", className="mt-4"),
                                            html.P(
                                                "We use a RandomForestClassifier with 100 trees. Random Forests handle complex data well "
                                                "and provide insights into which features matter most. Performance on the test set:",
                                                className="card-text text-muted",
                                                style={'fontSize': '1.1rem'}
                                            ),

                                            html.Ul(
                                                [
                                                    html.Li(f"Accuracy: {accuracy:.2%}"),
                                                    html.Li(f"F1 Score: {f1:.2%}"),
                                                    html.Li(f"Precision: {precision:.2%}"),
                                                    html.Li(f"Recall: {recall:.2%}")
                                                ],
                                                style={'fontSize': '1rem'}
                                            ),

                                            html.H5("Input Data for Prediction:", className="mt-3"),
                                            html.P(
                                                "Use the following inputs to see how our model classifies a new individual. "
                                                "Adjust each feature, then click 'Predict'.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            # GENDER
                                            html.Label("Gender:", style={'fontWeight': 'bold'}),
                                            dcc.Dropdown(
                                                id='input-gender',
                                                options=[
                                                    {'label': 'Male', 'value': 'Male'},
                                                    {'label': 'Female', 'value': 'Female'},
                                                ],
                                                placeholder='Select Gender',
                                                clearable=True,
                                                style={'marginBottom': '15px'}
                                            ),

                                            # EDUCATION
                                            html.Label("Education:", style={'fontWeight': 'bold'}),
                                            dcc.Dropdown(
                                                id='input-education',
                                                options=[
                                                            {'label': 'B.SC', 'value': 'B.SC'},
                                                            {'label': '12', 'value': '12'},
                                                            {'label': '8', 'value': '8'},
                                                            {'label': '6', 'value': '6'},
                                                            {'label': '10', 'value': '10'},
                                                            {'label': '0', 'value': '0'},
                                                            {'label': 'ITI', 'value': 'ITI'},
                                                            {'label': 'Diploma', 'value': 'Diploma'},
                                                            {'label': '11', 'value': '11'},
                                                            {'label': '9', 'value': '9'},
                                                            {'label': 'B.Com', 'value': 'B.Com'},
                                                            {'label': '7', 'value': '7'},
                                                            {'label': 'B.Sc', 'value': 'B.Sc'},
                                                            {'label': 'MCA', 'value': 'MCA'},
                                                            {'label': 'BA', 'value': 'BA'},
                                                            {'label': 'Bsc', 'value': 'Bsc'},
                                                            {'label': 'BA MA', 'value': 'BA MA'},
                                                            {'label': 'D.H', 'value': 'D.H'},
                                                            {'label': 'B.de', 'value': 'B.de'},
                                                            {'label': 'BBA', 'value': 'BBA'},
                                                            {'label': 'B.sc', 'value': 'B.sc'},
                                                            {'label': 'M.sc', 'value': 'M.sc'},
                                                            {'label': 'Msc', 'value': 'Msc'},
                                                            {'label': '5', 'value': '5'},
                                                            {'label': '3', 'value': '3'},
                                                            {'label': 'MA', 'value': 'MA'},
                                                            {'label': 'DM', 'value': 'DM'},
                                                            {'label': 'ug', 'value': 'ug'},
                                                            {'label': 'TEACHER', 'value': 'TEACHER'},
                                                            {'label': 'B.Ed', 'value': 'B.Ed'},
                                                            {'label': 'm.SC', 'value': 'm.SC'},
                                                            {'label': '4', 'value': '4'},
                                                            {'label': 'MSC BED', 'value': 'MSC BED'},
                                                            {'label': 'MBC', 'value': 'MBC'},
                                                            {'label': 'BC', 'value': 'BC'},
                                                            {'label': 'B.A', 'value': 'B.A'},
                                                            {'label': 'EEE', 'value': 'EEE'},
                                                            {'label': 'M.SC', 'value': 'M.SC'},
                                                            {'label': '15', 'value': '15'},
                                                            {'label': 'BSC', 'value': 'BSC'},
                                                            {'label': 'Diplomo', 'value': 'Diplomo'},
                                                            {'label': 'MSc', 'value': 'MSc'},
                                                            {'label': '1', 'value': '1'},
                                                            {'label': '2', 'value': '2'},
                                                            {'label': 'BA Bed', 'value': 'BA Bed'},
                                                            {'label': 'BSc Bed', 'value': 'BSc Bed'},
                                                            {'label': 'MA BED', 'value': 'MA BED'},
                                                            {'label': 'CS', 'value': 'CS'},
                                                            {'label': 'B.COM', 'value': 'B.COM'},
                                                        ],
                                                placeholder='Select Education',
                                                clearable=True,
                                                style={'marginBottom': '15px'}
                                            ),

                                            # MARITAL STATUS
                                            html.Label("Marital Status:", style={'fontWeight': 'bold'}),
                                            dcc.Dropdown(
                                                id='input-marital-status',
                                                options=[
                                                    {'label': 'Married', 'value': 'Married'},
                                                    {'label': 'Un Married', 'value': 'Un Married'},
                                                    {'label': 'Widow', 'value': 'Widow'},
                                                    {'label': 'Widower', 'value': 'Widower'},
                                                ],
                                                placeholder='Select Marital Status',
                                                clearable=True,
                                                style={'marginBottom': '15px'}
                                            ),

                                            # PRIMARY OCCUPATION
                                            html.Label("Primary Occupation:", style={'fontWeight': 'bold'}),
                                            dcc.Dropdown(
                                                id='input-primary-occupation',
                                                options=[
                                                            {'label': 'Agriculture', 'value': 'Agriculture'},
                                                            {'label': 'Cooly', 'value': 'Cooly'},
                                                            {'label': '0', 'value': '0'},
                                                            {'label': 'Pension', 'value': 'Pension'},
                                                            {'label': 'Security', 'value': 'Security'},
                                                            {'label': 'Driver', 'value': 'Driver'},
                                                            {'label': 'Post master (Rtd)', 'value': 'Post master (Rtd)'},
                                                            {'label': 'pension', 'value': 'pension'},
                                                            {'label': 'House wife', 'value': 'House wife'},
                                                            {'label': 'Student', 'value': 'Student'},
                                                            {'label': 'Research Assistant (tomporary)', 'value': 'Research Assistant (tomporary)'},
                                                            {'label': 'Gardner', 'value': 'Gardner'},
                                                            {'label': 'House Keeping', 'value': 'House Keeping'},
                                                            {'label': 'Petrol Bunk Pump Operater', 'value': 'Petrol Bunk Pump Operater'},
                                                            {'label': 'OAP', 'value': 'OAP'},
                                                            {'label': 'Private work', 'value': 'Private work'},
                                                            {'label': 'petti shop', 'value': 'petti shop'},
                                                            {'label': 'Tailer', 'value': 'Tailer'},
                                                            {'label': 'Shoes company', 'value': 'Shoes company'},
                                                            {'label': 'Superviser, Shoes company', 'value': 'Superviser, Shoes company'},
                                                            {'label': 'Depantment stor', 'value': 'Depantment stor'},
                                                            {'label': 'makeup', 'value': 'makeup'},
                                                            {'label': 'school work', 'value': 'school work'},
                                                            {'label': 'Mazon', 'value': 'Mazon'},
                                                            {'label': 'Agricultre', 'value': 'Agricultre'},
                                                            {'label': 'Painter', 'value': 'Painter'},
                                                            {'label': 'Teacher', 'value': 'Teacher'},
                                                            {'label': 'Wever', 'value': 'Wever'},
                                                            {'label': 'Small Business', 'value': 'Small Business'},
                                                            {'label': 'Work to school', 'value': 'Work to school'},
                                                            {'label': 'bank work', 'value': 'bank work'},
                                                            {'label': 'Beauty parlour', 'value': 'Beauty parlour'},
                                                            {'label': 'Un Employee', 'value': 'Un Employee'},
                                                            {'label': 'Beautician', 'value': 'Beautician'},
                                                            {'label': 'Watchman', 'value': 'Watchman'},
                                                        ],
                                                placeholder='Select Occupation',
                                                clearable=True,
                                                style={'marginBottom': '15px'}
                                            ),

                                            # TOTAL INCOME
                                            html.Label("Total Income:", style={'fontWeight': 'bold'}),
                                            dcc.Slider(
                                                id='input-total-income',
                                                min=0,
                                                max=100000,
                                                step=1000,
                                                value=20000,
                                                marks={
                                                    0: '0',
                                                    20000: '20k',
                                                    50000: '50k',
                                                    75000: '75k',
                                                    100000: '100k'
                                                },
                                                tooltip={"placement": "bottom", "always_visible": True},
                                                className='mb-3'
                                            ),

                                            html.Button(
                                                'Predict',
                                                id='predict-button',
                                                n_clicks=0,
                                                className='btn btn-primary'
                                            ),
                                            html.H5("Prediction Result:", className="mt-4"),
                                            html.Div(
                                                id='prediction-output',
                                                className="text-success font-weight-bold",
                                                style={'fontSize': '1.1rem'}
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInLeft",
                                style={'borderRadius': '15px', 'border': '1px solid #007bff', 'padding': '10px'}
                            ),
                            width=6
                        ),

                        # RIGHT: Explainable AI Components
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H4("Explainable AI Insights", className="text-primary")
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.H5("1. SHAP Summary Plot", className="mt-3"),
                                            html.P(
                                                "Shows how each feature impacts the model's prediction across all individuals. "
                                                "Red points indicate higher feature values, while blue points are lower feature values. "
                                                "Points to the right push predictions toward 'Vulnerable'; points to the left push them away.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            self.shap_summary.layout(),

                                            html.H5("2. Feature Importance", className="mt-4"),
                                            html.P(
                                                "Ranks features by how much they influence the overall predictions. A taller bar means "
                                                "that feature plays a bigger role in deciding vulnerability.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            self.feature_importance.layout(),

                                            html.H5("3. Dependence Plot", className="mt-4"),
                                            html.P(
                                                "Focuses on one feature (e.g., 'Gender_Male') to see how its changing value alters "
                                                "the model’s prediction. The color can represent another feature to highlight interactions.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            self.dependence.layout(),

                                            html.H5("4. Model Confusion Matrix", className="mt-4"),
                                            html.P(
                                                "Visualizes correct vs. incorrect classifications. Diagonal cells (True Positives, True Negatives) "
                                                "are correct predictions; off-diagonal cells (False Positives, False Negatives) show where the model errs.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            self.confusion_matrix.layout(),

                                            html.H5("5. Precision-Recall Curve", className="mt-4"),
                                            html.P(
                                                "Illustrates how well the model identifies actual 'Vulnerable' cases (Recall) "
                                                "while avoiding false alarms (Precision). Higher curves generally indicate better performance.",
                                                style={'fontSize': '1rem'}
                                            ),
                                            html.Img(
                                                src=f"data:image/png;base64,{self.pr_curve_img}",
                                                style={'width': '100%', 'marginBottom': '20px'}
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInRight",
                                style={'borderRadius': '15px', 'border': '1px solid #28a745', 'padding': '10px'}
                            ),
                            width=6
                        ),
                    ],
                    className="mt-4"
                ),

                # FOOTER
                dbc.Row(
                    dbc.Col(
                        html.Footer(
                            "\u00a9 2024 Vulnerability Prediction Dashboard | Data-Driven Insights & Explainability",
                            className="text-center text-light bg-dark p-3 mt-4 animate__animated animate__fadeInUp",
                            style={'fontFamily': 'Courier New, monospace', 'fontSize': '1rem'}
                        )
                    )
                )
            ],
            fluid=True,
            style={'backgroundColor': '#f8f9fa', 'padding': '20px'}
        )

##############################################################################
# 10. BUILD EXPLAINER DASHBOARD
##############################################################################
dashboard = ExplainerDashboard(
    explainer,
    CustomDashboard,
    title="AI FORA DASHBOARD",
    whatif=False,
    decision_trees=False,
    shap_interaction=False,
    simple=True
)

##############################################################################
# 11. CALLBACK FOR PREDICTION
##############################################################################
@dashboard.app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        Input('input-gender', 'value'),
        Input('input-education', 'value'),
        Input('input-marital-status', 'value'),
        Input('input-primary-occupation', 'value'),
        Input('input-total-income', 'value'),
    ]
)
def make_prediction(n_clicks, gender, education, marital_status, primary_occupation, total_income):
    if n_clicks > 0:
        # Handle None or blank
        gender = gender if gender else ""
        education = education if education else ""
        marital_status = marital_status if marital_status else ""
        primary_occupation = primary_occupation if primary_occupation else ""
        total_income = float(total_income) if total_income is not None else 0.0

        # Build a DataFrame with the same columns as training X
        input_data = {
            'Gender': gender,
            'Education': education,
            'Marital_status': marital_status,
            'Primary_occupation': primary_occupation,
            'Total_Income': total_income
        }
        input_df = pd.DataFrame([input_data])

        # Transform the categorical subset
        cat_df = input_df[cat_cols] if len(cat_cols) > 0 else pd.DataFrame()
        if len(cat_cols) > 0:
            cat_encoded = encoder.transform(cat_df)
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=sanitized_feature_names)
        else:
            cat_encoded_df = pd.DataFrame()

        numeric_input = input_df[numeric_cols] if len(numeric_cols) > 0 else pd.DataFrame()

        # Combine numeric + encoded
        input_final = pd.concat(
            [numeric_input.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)],
            axis=1
        )

        # Make the prediction
        pred = rf_classifier.predict(input_final)[0]
        prob = rf_classifier.predict_proba(input_final)[0, 1]

        result = "Vulnerable" if pred == 1 else "Not Vulnerable"
        return f"Prediction: {result} (Probability: {prob:.2%})"
    return ""

##############################################################################
# 12. FASTAPI SERVER
##############################################################################
server = FastAPI()
server.mount("/", WSGIMiddleware(dashboard.flask_server()))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:server", host="0.0.0.0", port=port, reload=True)
