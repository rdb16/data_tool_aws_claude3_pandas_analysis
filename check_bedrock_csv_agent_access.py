import pandas as pd
import boto3
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_aws.chat_models import ChatBedrock


def run_bedrock_agent(csv, model, profile_name="bedrock_user", region_name="eu-west-3"):
    """
    Active un agent Bedrock pour interagir avec un DataFrame Pandas.

    Args:
        csv (str): Chemin du fichier CSV.
        model (str): Identifiant du modèle Bedrock (e.g., anthropic.claude-3-haiku-20240307-v1:0).
        profile_name (str): Profil AWS dans le fichier credentials.
        region_name (str): Région AWS.

    Returns:
        None
    """
    # Charger le DataFrame à partir d'un CSV
    df_stocks = pd.read_csv(csv)

    # df_stocks = pd.read_csv()
    df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], utc=True)
    df_stocks['Date'] = df_stocks['Date'].dt.date

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
        df_stocks,
        verbose=True,
        allow_dangerous_code=True
    )

    # Exemples de requêtes à l'agent
    while True:
        user_input = input("Posez une question sur le dataframe (ou tapez 'exit' pour quitter) : ")
        if user_input.lower() == "exit":
            print("Fin de l'interaction.")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"Réponse de l'agent : {response}")
        except Exception as e:
            print(f"Erreur : {e}")


# Exemple d'utilisation
csv_path = "sources/World-Stock-Prices-Data-small.csv"
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
run_bedrock_agent(csv_path, model_id)
