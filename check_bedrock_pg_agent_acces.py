import psycopg2
from langchain_experimental.sql import SQLDatabase
from langchain_experimental.agents import create_sql_agent
from langchain_aws.chat_models import ChatBedrock
import boto3


def create_postgres_agent(
        model_id,
        host,
        database,
        user,
        password,
        profile_name="bedrock_user",
        region_name="eu-west-3"
):
    """
    Crée et retourne un agent SQL connecté à PostgreSQL et au modèle Bedrock.

    Args:
        model_id (str): Identifiant du modèle Bedrock (e.g., anthropic.claude-3-haiku-20240307-v1:0).
        host (str): Adresse du serveur PostgreSQL.
        database (str): Nom de la base de données PostgreSQL.
        user (str): Nom d'utilisateur PostgreSQL.
        password (str): Mot de passe PostgreSQL.
        profile_name (str): Profil AWS dans le fichier credentials.
        region_name (str): Région AWS.

    Returns:
        agent: Agent SQL configuré avec le modèle Bedrock.
    """
    # Créer une session Bedrock avec AWS
    session = boto3.Session(profile_name=profile_name)
    bedrock_llm = ChatBedrock(
        client=session.client("bedrock-runtime", region_name=region_name),
        model_id=model_id
    )

    # Connexion à la base de données PostgreSQL
    db_connection = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    sql_database = SQLDatabase.from_connection(db_connection)

    # Créer et retourner l'agent SQL
    agent = create_sql_agent(
        llm=bedrock_llm,
        db=sql_database,
        verbose=True,
        allow_dangerous_code=True  # Autorisation explicite pour exécuter du SQL dynamique
    )
    return agent


# Configuration de la base PostgreSQL
host = "your-postgres-host"
database = "your-database-name"
user = "your-username"
password = "your-password"

# Création de l'agent
model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # Modèle Bedrock à utiliser
postgres_agent = create_postgres_agent(
    model_id=model_id,
    host=host,
    database=database,
    user=user,
    password=password
)

# Gestion des prompts utilisateur
while True:
    user_input = input("Posez une question sur la base de données (ou tapez 'exit' pour quitter) : ")
    if user_input.lower() == "exit":
        print("Fin de l'interaction.")
        break

    try:
        # Utiliser invoke avec un dictionnaire en entrée
        response = postgres_agent.invoke({"input": user_input})
        print(f"Réponse de l'agent : {response}")
    except Exception as e:
        print(f"Erreur : {e}")
