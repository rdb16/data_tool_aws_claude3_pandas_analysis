from aws_bedrock_utils import get_body_from_simple_template, get_bedrock_client
import os
import json

if not os.path.exists('./tmp'):
    print("tmp was created.")
    os.mkdir('./tmp')
elif os.path.exists('./tmp/control.json'):
    os.remove('./tmp/control.json')

print("Path is clean, we start !!")

question = "What could you tell us about quantum computing ?. Please, make a resume in french."
payload = get_body_from_simple_template(question)
with open('./tmp/control.json', 'w') as f:
    json.dump(payload, f, indent=4)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
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


