from langchain_aws import BedrockLLM


custom_llm = BedrockLLM(
    credentials_profile_name="bedrock_user",
    provider="anthropic",
    region="eu-west-3",
    model_id="arn:aws:bedrock:eu-west-3::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0},
    streaming=True
)

custom_llm.invoke(input="Quelle est la recette de la mayonnaise ?")