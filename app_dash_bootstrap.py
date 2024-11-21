import pandas as pd
from dash import Dash, html, Input, Output, dash_table, dcc, callback
import plotly.express as px
import dash_bootstrap_components as dbc

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    dbc.Row([
        html.Div('My First APP with Data, Graph and Controls', className='text-primary text-center fs-3')
    ]),
    dbc.Row([
        dbc.RadioItems(options=[{"label": x, "value": x} for x in ['pop', 'lifeExp', 'gdpPercap']],
                       value='lifeExp',
                       inline=True,
                       id='radio-buttons-final'
                       )
    ]),
    dbc.Row([
        dbc.Col([
            dash_table.DataTable(data=df.to_dict('records'), page_size=12, style_table={'overflowX': 'auto'})
        ], width=6),
        dbc.Col([
            dcc.Graph(figure={}, id='my-first-graph-final')
        ], width=6),
    ]),
], fluid=True)


# Ajour d'un callback pour construire l'interaction
@callback(
    Output(component_id='my-first-graph-final', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
