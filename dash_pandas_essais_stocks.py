import time
import uuid
from fpdf import FPDF
import json
import boto3
from dash import Dash, dcc, html, callback, Input, Output, State, no_update, callback_context, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from langchain_aws import ChatBedrock
from langchain_experimental.agents import create_pandas_dataframe_agent
import datetime
import os
from selenium import webdriver


def run_bedrock_agent(dfr, model, question, profile_name="bedrock_user", region_name="eu-west-3"):
    """
    Active un agent Bedrock pour interagir avec un DataFrame Pandas.

    Args:
        dfr (pd.DataFrame): Pandas DataFrame avec le quel interagir.
        model (str): Identifiant du modèle Bedrock (e.g., anthropic.claude-3-haiku-20240307-v1:0).
        profile_name (str): Profil AWS dans le fichier credentials.
        region_name (str): Région AWS.

    Returns:
        None
    """
    # Créer une session Bedrock avec AWS
    session = boto3.Session(profile_name=profile_name)
    bedrock_llm = ChatBedrock(
        client=session.client("bedrock-runtime", region_name=region_name),
        model_id=model,
        max_tokens=600,
    )

    # Créer l'agent Pandas avec LangChain
    agent = create_pandas_dataframe_agent(
        bedrock_llm,
        dfr,
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )

    response = None

    try:
        response = agent.invoke({"input": question})
        print(f"Réponse de l'agent : {response['output']}")
    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        return response['output']


# Fonction pour ajouter une question/réponse au fichier JSON
def append_to_session_state(question, answer):
    with open(session_json_file, "r") as f:
        session_state = json.load(f)
    session_state["questions"].append(question)
    session_state["answers"].append(answer)
    with open(session_json_file, "w") as f:
        json.dump(session_state, f)


class CustomPDF(FPDF):
    def footer(self):
        # à 15mm du bas
        self.set_y(-16)

        # Libellé centré
        self.set_font("Arial", "B", 8)
        self.set_text_color(30,30,30)
        # réduire l'interligne à 5 (h)
        self.cell(0, 5, "Powered by SNTP Capitalisation", ln=True, align="C")

        # Copyrights centré
        self.set_y(-10)
        self.set_font("Arial", size=8)
        self.cell(0, 5, "© 2024 All rights reserved.", align="C")
        self.set_text_color(0,0,0)


# Fonction pour générer un PDF à partir des données JSON
def generate_pdf_from_session(csv, session_file):
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    csv_basename = os.path.basename(csv)
    pdf_file = f"./results/{os.path.splitext(csv_basename)[0]}_{timestamp}.pdf"
    # debug
    # print(pdf_file)

    # Charger le contenu JSON
    with open(session_file, "r") as f:
        session_state = json.load(f)

    # Créer le PDF
    pdf = CustomPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # logo centré
    logo_path = "./img/armoiries-ecu.jpg"
    pdf.image(logo_path, x=(210 - 30) / 2, y=10, w=30)  # centré largeur 50mm

    # Ajouter la date du jour
    pdf.set_y(31)  # Position sous le logo
    pdf.set_x(150)  # position horizontale
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Date : {datetime.datetime.now().strftime('%d/%m/%Y')}", ln=True, align="R")

    # Ajouter le nom du fichier CSV dans un encadré
    pdf.ln(5)  # Espacement
    pdf.set_font("Arial", style="B", size=12)
    pdf.set_fill_color(0, 0, 128)  # Bleu marine
    pdf.set_text_color(255, 255, 255)  # Blanc
    pdf.cell(0, 10, txt=csv_basename, border=1, ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)  # Retour à la couleur noire par défaut

    # centré le titre
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt="Session Questions/Réponses", ln=True, align="C")
    pdf.ln(10)  # Espacement

    # Ajouter les questions et réponses
    for i, (question, answer) in enumerate(zip(session_state["questions"], session_state["answers"]), start=1):
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt=f"Question {i}:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=question)
        pdf.ln(5)
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(0, 10, txt=f"Réponse {i}:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=answer)
        pdf.ln(10)

    # Sauvegarder le PDF
    pdf.output(pdf_file)
    return pdf_file


# Initialisation de l'application avec les feuilles de style de Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

# Initialisation de l'environnement temporaire
TMP_DIR = "./tmp"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# Générer un UUID unique pour chaque utilisateur
user_uuid = str(uuid.uuid4())
session_json_file = os.path.join(TMP_DIR, f"session_state_{user_uuid}.json")

# Créer un fichier JSON vierge pour la session
if os.path.exists(session_json_file):
    os.remove(session_json_file)
with open(session_json_file, "w") as f:
    json.dump({"questions": [], "answers": []}, f)

# Chargement des données
csv_file = 'sources/World-Stock-Prices-Data-small.csv'
df_stocks = pd.read_csv(csv_file)
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], utc=True)
df_stocks['Date'] = df_stocks['Date'].dt.date

# Layout principal
app.layout = dbc.Container([
    dcc.Markdown("# Analyse de données sur les actions", className='mb-4 mt-4',
                 style={'text-align': 'center', 'color': 'blue'}),

    # Graphique en haut
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='stock-picker',
                options=[{'label': ticker, 'value': ticker} for ticker in sorted(df_stocks['Ticker'].unique())],
                value=['AAPL', 'TSLA'],
                multi=True,
                className="mb-2"
            ),
            dcc.Graph(id='line-chart')
        ], width=12)
    ], className='mb-4'),

    # Section dynamique pour les questions
    dbc.Row([
        dbc.Col([
            dbc.Button("Poser une question", id="btn-question", color="success", className="me-2"),
            dbc.Button("Arrêter", id="btn-stop", color="danger", className="me-2"),
            html.Div(id="dynamic-section", className="mt-2")
        ], width=12)
    ]),
    # Footer
    html.Footer(
        [
            html.Div("Powered by SNTP Capitalisation", className="text-center", style={"font-weight": "bold"}),
            html.Div("© 2024 All rights reserved.", className="text-center", style={"font-size": "smaller"})
        ],
        style={
            "position": "fixed",  # Toujours visible
            "bottom": "0",
            "left": "0",
            "width": "100%",
            "background-color": "#f8f9fa",  # Couleur claire
            "padding": "10px 0",
            "border-top": "1px solid #ddd",
            "text-align": "center",
        }
    )
])




# Callback pour mettre à jour le graphique
@app.callback(
    Output("line-chart", "figure"),
    Input("stock-picker", "value")
)
def update_graph(selected_stocks):
    df = df_stocks[df_stocks['Ticker'].isin(selected_stocks)]
    figure = px.line(df, x='Date', y='Close', color='Ticker', title="Cours des actions")
    return figure


# Callback combiné pour gérer le bouton "Poser une question" et "Arrêter"
@app.callback(
    Output("dynamic-section", "children"),
    [Input("btn-question", "n_clicks"), Input("btn-stop", "n_clicks")],
    State("dynamic-section", "children"),
    prevent_initial_call=True
)
def manage_dynamic_section(btn_question, btn_stop, children):
    # Initialiser `children` s'il est vide
    if children is None:
        children = []

    # Déterminer quel bouton a été cliqué
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "btn-question":
        # Ajouter une nouvelle ligne avec une question
        n_clicks = len(children)  # Utiliser la longueur actuelle comme index
        question_row = dbc.Row(
            [
                dbc.Col(dbc.Input(id={"type": "question-input", "index": n_clicks},
                                  placeholder="Posez une question",
                                  type="text",
                                  style={"word-wrap": "break-word", "width": "100%"},
                                  autofocus=True
                                  ),
                        width=8),
                dbc.Col(
                    dbc.Button(
                        "Soumettre",
                        id={"type": "submit-button", "index": n_clicks},
                        color="primary",
                        style={"width": "100%"}
                    ),
                    width=2),
                dbc.Col(
                    html.Div(
                        id={"type": "answer-output", "index": n_clicks},
                        style={"word-wrap": "break-word", "width": "100%"}
                    ), width=8)
            ],
            className="mb-2 align-items-center",
        )
        return children + [question_row]

    elif triggered_id == "btn-stop":
        # Enregistrer l'état actuel dans un fichier PDF
        pdf_file = generate_pdf_from_session(csv_file, session_json_file)

        # Réinitialiser la section dynamique
        return f"Session sauvegardée dans {pdf_file}"

    return children


# Callback pour gérer la soumission des questions et retirer le bouton "Soumettre"
@app.callback(
    Output({"type": "answer-output", "index": MATCH}, "children"),
    Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
    State({"type": "question-input", "index": MATCH}, "value"),
    prevent_initial_call=True
)
def handle_question_submission(n_clicks, question):
    if not question:
        return html.Div("Erreur : aucune question spécifiée", className="alert alert-danger")

    # Appeler le modèle pour obtenir une réponse
    response = run_bedrock_agent(df_stocks, "anthropic.claude-3-haiku-20240307-v1:0", question)

    # Sauvegarder dans le JSON
    append_to_session_state(question, response)

    return html.Div(f"Réponse : {response}", className="alert alert-info")


# Callback pour supprimer le bouton "Soumettre" après l'affichage de la réponse
@app.callback(
    Output({"type": "submit-button", "index": MATCH}, "style"),
    Input({"type": "submit-button", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def remove_submit_button(n_clicks):
    return {"display": "none"}


if __name__ == '__main__':
    app.run_server(debug=True)
