# import os
# import sys 
# import json

# if "LAMBDA_TASK_ROOT" in os.environ:
#     envLambdaTaskRoot = os.environ["LAMBDA_TASK_ROOT"]
#     sys.path.insert(0, "/var/lang/lib/python3.9/site-packages")

# from langchain.retrievers import AmazonKendraRetriever
# from langchain.chains import ConversationalRetrievalChain
# from langchain.llms.bedrock import Bedrock
# from langchain.prompts import PromptTemplate

# REGION_NAME = os.environ['aws_region']

# MODEL_TYPE = "AMAZON"

# retriever = AmazonKendraRetriever(
#     index_id=os.environ['kendra_index_id'],
#     region_name=REGION_NAME
# )

# if(MODEL_TYPE == "CLAUDE"):
#     llm = Bedrock(
#         model_id="anthropic.claude-3-haiku-20240307-v1:0",
#         endpoint_url="https://bedrock-runtime." + REGION_NAME + ".amazonaws.com",
#         model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 500}
#     )

#     condense_question_llm = Bedrock(
#         model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#         endpoint_url="https://bedrock-runtime." + REGION_NAME + ".amazonaws.com",
#         model_kwargs={"temperature": 0.7, "max_tokens_to_sample": 300}
#     )
# else:
#     llm = Bedrock(
#         model_id="amazon.titan-text-lite-v1",
#         endpoint_url="https://bedrock-runtime." + REGION_NAME + ".amazonaws.com",
#         model_kwargs={"temperature": 0.7, "maxTokens": 500, "numResults": 1}
#     )

#     condense_question_llm = Bedrock(
#         model_id="amazon.titan-text-lite-v1",
#         endpoint_url="https://bedrock-runtime." + REGION_NAME + ".amazonaws.com",
#         model_kwargs={"temperature": 0.7, "maxTokens": 300, "numResults": 1}
#     )

# #Create template for combining chat history and follow up question into a standalone question.
# question_generator_chain_template = """
# Human: Here is some chat history contained in the <chat_history> tags. If relevant, add context from the Human's previous questions to the new question. Return only the question. No preamble. If unsure, ask the Human to clarify. Think step by step.

# Assistant: Ok

# <chat_history>
# {chat_history}

# Human: {question}
# </chat_history>

# Assistant:
# """

# question_generator_chain_prompt = PromptTemplate.from_template(question_generator_chain_template)

# #Create template for asking the question of the given context.
# combine_docs_chain_template = """
# Human: You are a friendly, concise chatbot. Here is some context, contained in <context> tags. Answer this question as concisely as possible with no tags. Say I don't know if the answer isn't given in the context: {question}

# <context>
# {context}
# </context>

# Assistant:
# """
# combine_docs_chain_prompt = PromptTemplate.from_template(combine_docs_chain_template)

# # RetrievalQA instance with custom prompt template
# qa = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     condense_question_llm=condense_question_llm,
#     retriever=retriever,
#     return_source_documents=True,
#     condense_question_prompt=question_generator_chain_prompt,
#     combine_docs_chain_kwargs={"prompt": combine_docs_chain_prompt}
# )

# # This function handles formatting responses back to Lex.
# def lex_format_response(event, response_text, chat_history):
#     event['sessionState']['intent']['state'] = "Fulfilled"

#     return {
#         'sessionState': {
#             'sessionAttributes': {'chat_history': chat_history},
#             'dialogAction': {
#                 'type': 'Close'
#             },
#             'intent': event['sessionState']['intent']
#         },
#         'messages': [{'contentType': 'PlainText','content': response_text}],
#         'sessionId': event['sessionId'],
#         'requestAttributes': event['requestAttributes'] if 'requestAttributes' in event else None
#     }

# def lambda_handler(event, context):
#     if(event['inputTranscript']):
#         user_input = event['inputTranscript']
#         prev_session = event['sessionState']['sessionAttributes']

#         print(prev_session)

#         # Load chat history from previous session.
#         if 'chat_history' in prev_session:
#             chat_history = list(tuple(pair) for pair in json.loads(prev_session['chat_history']))
#         else:
#             chat_history = []

#         if user_input.strip() == "":
#             result = {"answer": "Please provide a question."}
#         else:
#             input_variables = {
#                 "question": user_input,
#                 "chat_history": chat_history
#             }

#             print(f"Input variables: {input_variables}")

#             result = qa(input_variables)

#         # If Kendra doesn't return any relevant documents, then hard code the response 
#         # as an added protection from hallucinations.
#         if(len(result['source_documents']) > 0):
#             response_text = result["answer"].strip() 
#         else:
#             response_text = "I don't know"

#         # Append user input and response to chat history. Then only retain last 3 message histories.
#         # It seemed to work better with AI responses removed, but try adding them back in. {response_text}
#         chat_history.append((f"{user_input}", f"..."))
#         chat_history = chat_history[-3:]

#         return lex_format_response(event, response_text, json.dumps(chat_history))


import json
import boto3
import os
import sys
import logging
from botocore.exceptions import ClientError

LOG = logging.getLogger()
LOG.setLevel(logging.INFO)
if "LAMBDA_TASK_ROOT" in os.environ:
    envLambdaTaskRoot = os.environ["LAMBDA_TASK_ROOT"]
    sys.path.insert(0, "/var/lang/lib/python3.9/site-packages")

region_name = os.environ['aws_region']
s3_bucket = os.getenv("bucket")
model_id = os.getenv("model_id", "amazon.titan-text-express-v1")

# Bedrock client used to interact with APIs around models
bedrock = boto3.client(service_name="bedrock", region_name=region_name)

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region_name)


def get_session_attributes(intent_request):
    session_state = intent_request["sessionState"]
    if "sessionAttributes" in session_state:
        return session_state["sessionAttributes"]

    return {}


def close(intent_request, session_attributes, fulfillment_state, message):
    intent_request["sessionState"]["intent"]["state"] = fulfillment_state
    return {
        "sessionState": {
            "sessionAttributes": session_attributes,
            "dialogAction": {"type": "Close"},
            "intent": intent_request["sessionState"]["intent"],
        },
        "messages": [message],
        "sessionId": intent_request["sessionId"],
        "requestAttributes": intent_request["requestAttributes"]
        if "requestAttributes" in intent_request
        else None,
    }


def lambda_handler(event, context):
    LOG.info(f"Event is {event}")
    accept = "application/json"
    content_type = "application/json"
    prompt = event["inputTranscript"]

    try:
        request = json.dumps({
             "inputText": "\n\nHuman:" + prompt + "\n\nAssistant:",
        })
        # request = json.dumps(
        #     {
        #         "prompt": "\n\nHuman:" + prompt + "\n\nAssistant:",
        #         "max_tokens_to_sample": 4096,
        #         "temperature": 0.5,
        #         "top_k": 250,
        #         "top_p": 1,
        #         "stop_sequences": ["\\n\\nHuman:"],
        #     }
        # )
        
        response = bedrock_runtime.invoke_model(
            body=request,
            modelId=model_id,
            accept=accept,
            contentType=content_type,
        )
        LOG.info(f"Response: {response}")
        response_body = json.loads(response["body"].read().decode("utf-8"))
        LOG.info(f"Response body: {response_body}")
        message = response_body["results"][0]["outputText"]
        response_message = {
            "contentType": "PlainText",
            "content": message,
        }
        session_attributes = get_session_attributes(event)
        fulfillment_state = "Fulfilled"
        return close(event, session_attributes, fulfillment_state, response_message)
    
    except ClientError as e:
        LOG.error(f"Exception raised while execution and the error is {e}")
