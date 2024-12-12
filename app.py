import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import *
import dash_bootstrap_components as dbc
from dash import dcc, html
import matplotlib.pyplot as plt
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn
import io
import base64

# Load the data
data = pd.read_csv("pdsvul.csv")
data.columns = data.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
data['Education'] = data['Education'].str.replace(r'[^a-zA-Z0-9_ ]', '', regex=True)
data['Education'] = data['Education'].str.strip()
data['Education'] = data['Education'].str.upper()
data['Vulnerable'] = data['Vulnerable'].str.lower().replace({'yes': 1, 'no': 0})

# Prepare the data
X = data.drop(columns=['Vulnerable'])
y = data['Vulnerable'].astype(int)
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object']))
encoded_feature_names = encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns)
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
X_final = pd.concat([X.select_dtypes(exclude=['object']), X_encoded], axis=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Model Performance Metrics
y_pred = rf_classifier.predict(X_test)
y_prob = rf_classifier.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Create an explainer object for the dashboard
explainer = ClassifierExplainer(rf_classifier, X_test, y_test)

# Create a precision-recall curve plot
def create_precision_recall_plot():
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, marker='.', label='RandomForest')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Custom dashboard using explainerdashboard components
class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, name=name)
        self.shap_summary = ShapSummaryComponent(explainer)
        self.feature_importance = ImportancesComponent(explainer)
        self.dependence = ShapDependenceComponent(explainer, col="Gender_Male")
        self.confusion_matrix = ConfusionMatrixComponent(explainer)
        self.feature_input = FeatureInputComponent(explainer)
        self.pr_curve_img = create_precision_recall_plot()

    def layout(self):
        return dbc.Container(
            [
                # Split into two halves: Model Info on the left, Explainability on the right
                dbc.Row(
                    [
                        # Left Half: Model Info, Inputs, and Prediction Result
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H4("Model Overview, Inputs, and Prediction", className="text-primary")),
                                    dbc.CardBody(
                                        [
                                            html.H5("Classification Problem", className="mt-3"),
                                            html.P(
                                                "The classification problem aims to predict whether an individual is considered vulnerable based on socio-economic factors such as education, occupation, total income, and gender.",
                                                className="card-text text-muted",
                                                style={'fontSize': '1.1rem'}
                                            ),
                                            html.H5("Model Information", className="mt-4"),
                                            html.P(
                                                "Model: RandomForestClassifier with 100 trees, suited for handling feature interactions and providing feature importance metrics.",
                                                className="card-text text-muted",
                                                style={'fontSize': '1.1rem'}
                                            ),
                                            html.H5("Performance Metrics:", className="mt-3"),
                                            html.Ul(
                                                [
                                                    html.Li(f"Accuracy: {accuracy:.2%}", className="text-muted"),
                                                    html.Li(f"F1 Score: {f1:.2%}", className="text-muted"),
                                                    html.Li(f"Precision: {precision:.2%}", className="text-muted"),
                                                    html.Li(f"Recall: {recall:.2%}", className="text-muted"),
                                                ],
                                                style={'fontSize': '1rem'}
                                            ),
                                            html.H5("Input Data for Prediction:", className="mt-3"),
                                            # Input fields for each feature
                                            html.Div(
                                                [
                                                    dcc.Input(id='input-education', type='text', placeholder='Education', className='mb-2'),
                                                    dcc.Input(id='input-occupation', type='text', placeholder='Occupation', className='mb-2'),
                                                    dcc.Input(id='input-income', type='number', placeholder='Total Income', className='mb-2'),
                                                    dcc.Input(id='input-gender', type='text', placeholder='Gender', className='mb-2'),
                                                ],
                                                className="mb-3"
                                            ),
                                            html.Button('Predict', id='predict-button', n_clicks=0, className='btn btn-primary'),
                                            html.H5("Prediction Result:", className="mt-4"),
                                            html.Div(id='prediction-output', className="text-success font-weight-bold")
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInLeft",
                                style={'borderRadius': '15px', 'border': '1px solid #007bff'}
                            ),
                            width=6
                        ),
                        
                        # Right Half: Explainable AI Components
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H4("Explainable AI Insights", className="text-primary")),
                                    dbc.CardBody(
                                        [
                                            html.H5("SHAP Summary Plot", className="mt-3"),
                                            self.shap_summary.layout(),
                                            html.H5("Feature Importance", className="mt-4"),
                                            self.feature_importance.layout(),
                                            html.H5(f"Dependence Plot for 'Gender_Male'", className="mt-4"),
                                            self.dependence.layout(),
                                            html.H5("Model Confusion Matrix", className="mt-4"),
                                            self.confusion_matrix.layout(),
                                            html.H5("Precision-Recall Curve", className="mt-4"),
                                            html.Img(src=f"data:image/png;base64,{self.pr_curve_img}", style={'width': '100%'}),
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInRight",
                                style={'borderRadius': '15px', 'border': '1px solid #28a745'}
                            ),
                            width=6
                        ),
                    ],
                    className="mt-4"
                ),
                
                # Footer Section
                dbc.Row(
                    dbc.Col(
                        html.Footer(
                            "© 2024 Vulnerability Prediction Dashboard | Data-Driven Insights & Explainability",
                            className="text-center text-light bg-dark p-3 mt-4 animate__animated animate__fadeInUp",
                            style={'fontFamily': 'Courier New, monospace', 'fontSize': '1rem'}
                        )
                    )
                )
            ],
            fluid=True,
            style={'backgroundColor': '#f8f9fa', 'padding': '20px'}
        )

# Initialize and run the dashboard
dashboard = ExplainerDashboard(
    explainer,
    CustomDashboard,
    title="AI FORA DASHBOARD",
    whatif=False,
    decision_trees=False,
    shap_interaction=False,
    simple=True
)

# Create FastAPI app
server = FastAPI()
server.mount("/", WSGIMiddleware(dashboard.flask_server()))

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
