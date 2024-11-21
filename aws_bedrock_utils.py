import json
import pandas as pd
import boto3
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_aws.chat_models import ChatBedrock


def get_bedrock_client(user):
    # Initialiser une session AWS en utilisant le profil spécifié
    session = boto3.Session(profile_name=user)
    # Créer un client pour AWS Bedrock
    bdr_client = session.client('bedrock-runtime', region_name='eu-west-3')
    # le corps de la requête
    return bdr_client


def get_body_from_simple_template(question):
    with open('templates/body_simple_template.json', 'r') as file:
        data = json.load(file)
        if data['messages'][0]['content'][0]['text'] == "":
            data['messages'][0]['content'][0]['text'] = question

        return data


def get_body_from_data_analysis_template(question):
    with open('templates/body_data_analysis_template.json', 'r') as file:
        data = json.load(file)
        if data['messages'][1]['content'][0]['text'] == "":
            data['messages'][1]['content'][0]['text'] = question

        return data


def get_bedrock_df_agent(dfr, model, profile_name="bedrock_user", region_name="eu-west-3"):
    """
    Active un agent Bedrock pour interagir avec un DataFrame Pandas.

    Args:
        dfr (pd.DataFrame): Pandas DataFrame avec le quel interagir.
        model (str): Identifiant du modèle Bedrock (e.g., anthropic.claude-3-haiku-20240307-v1:0).
        profile_name (str): Profil AWS dans le fichier credentials.
        region_name (str): Région AWS.

    Returns:
        agent: un Agent Pandas configuré avec le service aws Bedrock
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
    return agent
