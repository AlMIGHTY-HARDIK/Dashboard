import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.custom import *
import dash_bootstrap_components as dbc
from dash import dcc, html
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
import uvicorn

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
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Create an explainer object for the dashboard
explainer = ClassifierExplainer(rf_classifier, X_test, y_test)

# Custom dashboard using explainerdashboard components
class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, name=name)
        self.shap_summary = ShapSummaryComponent(explainer)
        self.feature_importance = ImportancesComponent(explainer)
        self.dependence = ShapDependenceComponent(explainer, col="Gender_Male")

    def layout(self):
        return dbc.Container(
            [
                # Split into two halves: Model Info on the left, Explainability on the right
                dbc.Row(
                    [
                        # Left Half: Model Info and Problem Description
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H4("Model Overview and Classification Problem", className="text-primary")),
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
                                                    html.Li("Precision, Recall: Calculated for class balance", className="text-muted"),
                                                ],
                                                style={'fontSize': '1rem'}
                                            ),
                                            html.H5("Inputs:", className="mt-3"),
                                            html.P(
                                                "Key inputs include socio-economic indicators like education levels, job types, and income ranges, allowing the model to learn patterns related to vulnerability.",
                                                className="card-text text-muted",
                                                style={'fontSize': '1rem'}
                                            ),
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInLeft",
                                style={'borderRadius': '15px', 'border': '1px solid #007bff'}
                            ),
                            width=6  # Takes 50% of the page width
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
                                        ]
                                    ),
                                ],
                                className="shadow-sm animate__animated animate__fadeInRight",
                                style={'borderRadius': '15px', 'border': '1px solid #28a745'}
                            ),
                            width=6  # Takes 50% of the page width
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

if __name__ == '__main__':
    uvicorn.run("app:server", host="127.0.0.1", port=8000, reload=True)
