from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import os
import boto3
import json


def get_bedrock_client(user):
    # Initialiser une session AWS en utilisant le profil spécifié
    session = boto3.Session(profile_name=user)
    # Créer un client pour AWS Bedrock
    bdr_client = session.client('bedrock-runtime', region_name='eu-west-3')
    # le corps de la requête
    return bdr_client


def get_body_from_template(question):
    with open('templates/body_template.json', 'r') as file:
        data = json.load(file)
        if data['messages'][0]['content'][0]['text'] == "":
            data['messages'][0]['content'][0]['text'] = question

        return data


if __name__ == '__main__':
    # Exemple de requête pour appeler le modèle spécifique
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    payload = get_body_from_template("What could you tell us about quantum computing?")
    body = json.dumps(payload).encode('utf-8')
    bedrock_client = get_bedrock_client("bedrock_user")

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        response_body = json.loads(response['body'].read())
        print(response_body['content'][0]['text'])
    except Exception as e:
        print(f"Erreur lors de l'invocation du modèle : {e}")


