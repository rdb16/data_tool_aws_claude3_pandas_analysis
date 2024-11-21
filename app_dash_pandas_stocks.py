import boto3
from dash.exceptions import PreventUpdate
from langchain_aws import ChatBedrock
from langchain_experimental.agents import create_pandas_dataframe_agent
import pdfkit
from dash import Dash, html, dcc, callback, Input, Output, State, MATCH
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import uuid


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
        allow_dangerous_code=True
    )

    response = None

    try:
        response = agent.invoke({"input": question})
        print(f"Réponse de l'agent : {response['output']}")
    except Exception as e:
        print(f"Erreur : {e}")
    finally:
        return response['output']


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

df_stocks = pd.read_csv('sources/World-Stock-Prices-Data-small.csv')
df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], utc=True)
df_stocks['Date'] = df_stocks['Date'].dt.date

app.layout = dbc.Container([
    dcc.Markdown("# Analyse de données sur les actions", className='mb-4 mt-4'),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='stock-picker',
                options=[{'label': ticker, 'value': ticker} for ticker in sorted(df_stocks['Ticker'].unique())],
                value=['AAPL', 'TSLA'],
                multi=True
            )
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-chart')
        ], width=12)
    ]),
    html.Div(id='interaction-container', children=[
        dbc.InputGroup([
            dbc.Input(id='input-question', type='text', placeholder='Posez votre question ici'),
            dbc.Button('Poser la question', id='ask-button', n_clicks=0, color='primary', className='ms-2'),
        ], className='mb-3')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Poser une autre question', id='ask-again-button', n_clicks=0, color='success', className='me-2',
                       style={'display': 'block'}),
            dbc.Button('Arrêter et sauvegarder', id='save-button', n_clicks=0, color='danger',
                       style={'display': 'none'})
        ], className='d-flex justify-content-end')
    ], className='mt-3')
], fluid=True)


@app.callback(
    [Output('interaction-container', 'children'),
     Output('ask-again-button', 'style'),
     Output('save-button', 'style')],
    [Input('ask-button', 'n_clicks'),
     Input('ask-again-button', 'n_clicks')],
    [State('interaction-container', 'children'),
     State('input-question', 'value')],
    prevent_initial_call=True
)
def handle_interaction(ask_clicks, again_clicks, children, question):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if 'ask-button' in triggered_id:
        if ask_clicks > 0:
            response = f"Fake response for demonstration: {question}"
            response_display = dbc.InputGroup(
                [dbc.Input(value=f"Réponse : {response}", plaintext=True, readonly=True, className='mb-3')],
                className='mb-3'
            )
            children.append(response_display)
            return children, {'display': 'block'}, {'display': 'block'}
    elif 'ask-again-button' in triggered_id:
        if again_clicks > 0:
            new_input_group = html.Div([
                dbc.InputGroup([
                    dbc.Input(id={'type': 'input-question', 'index': again_clicks}, type='text',
                              placeholder='Posez votre question ici'),
                    dbc.Button('Poser la question', id={'type': 'ask-button', 'index': again_clicks}, n_clicks=0,
                               color='primary'),
                ], className='mb-3'),
                html.Div(id={'type': 'answer-space', 'index': again_clicks})
            ])
            children.append(new_input_group)
            return children, {'display': 'block'}, {'display': 'block'}
    return children, {'display': 'none'}, {'display': 'none'}


if __name__ == '__main__':
    app.run_server(debug=True)


