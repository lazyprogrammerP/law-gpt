import os

import dotenv
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2
from fastapi import FastAPI, UploadFile, File
from langdetect import detect

from helpers.documents.load_document import load_document
from helpers.pinecone.get_relevant_documents import get_relevant_documents
from helpers.pinecone.get_vector_store import get_vector_store
from helpers.translators.english_to_hindi import english_to_hindi
from helpers.translators.english_to_tamil import english_to_tamil
from helpers.translators.hindi_to_english import hindi_to_english
from helpers.translators.tamil_to_english import tamil_to_english

dotenv.load_dotenv()

app = FastAPI()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

LLAMA_USER_ID = "meta"
LLAMA_APP_ID = "Llama-2"
LLAMA_MODEL_ID = "llama2-70b-chat"
LLAMA_MODEL_VERSION_ID = "6c27e86364ba461d98de95cddc559cb3"

WHISPER_USER_ID = "openai"
WHISPER_APP_ID = "transcription"
WHISPER_MODEL_ID = "whisper"
WHISPER_MODEL_VERSION_ID = "ccfd40cc37c448ef87fd5f166e7cb16e"


@app.post("/api/v1/vectorize")
def vectorize(document: UploadFile = File(...)):
    document_chunks, document_path = load_document(document=document)

    vector_store = get_vector_store()
    vector_store.add_documents(documents=document_chunks)

    return {"message": "Created the vectors successfully."}


@app.get("/api/v1/text/assist")
def text_assist(prompt: str):
    language = detect(text=prompt)

    if language == "hi":
        prompt = hindi_to_english(hindi_text=prompt)

    if language == "ta":
        prompt = tamil_to_english(tamil_text=prompt)

    relevant_documents = get_relevant_documents(query=prompt)

    context = ""
    for relevant_document in relevant_documents:
        context += f"{relevant_document.page_content}\n"

    llama_instruction = f"""
    <s>
    [INST]
    <<SYS>>
    You are a highly skilled, professional and respectful lawyer.
    Given the text extracts from the Indian Constitution, your job is to provide legally sound and unbiased answers.
    Explain incoherent questions. Refrain from sharing false information.
    You are expected to address hypothetical scenarios with constitutional context.
    Try to answer as briefly as possible.
    Remember! The answer should only pertain to the given context.
    <</SYS>>
    
    ### BEGIN CONTEXT
    {context}
    ### END CONTEXT
    
    Question: {prompt}
    [/INST]
    Answer: 
    """

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + CLARIFAI_PAT),)
    user_data_object = resources_pb2.UserAppIDSet(user_id=LLAMA_USER_ID, app_id=LLAMA_APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_data_object,
            model_id=LLAMA_MODEL_ID,
            version_id=LLAMA_MODEL_VERSION_ID,
            inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=llama_instruction)))]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        return post_model_outputs_response
    output_choice = post_model_outputs_response.outputs[0]

    if output_choice is None:
        return None

    output = output_choice.data.text.raw.strip()

    if language == "hi":
        output = english_to_hindi(english_text=output)

    if language == "ta":
        prompt = english_to_tamil(english_text=prompt)

    return {"message": output}


@app.get("/api/v1/audio/assist")
def audio_assist(audio_url: str):
    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + CLARIFAI_PAT),)

    user_data_object = resources_pb2.UserAppIDSet(user_id=WHISPER_USER_ID, app_id=WHISPER_APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_data_object,
            model_id=WHISPER_MODEL_ID,
            version_id=WHISPER_MODEL_VERSION_ID,
            inputs=[resources_pb2.Input(data=resources_pb2.Data(audio=resources_pb2.Audio(url=audio_url)))]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        return post_model_outputs_response

    audio_transcript = post_model_outputs_response.outputs[0].data.text.raw

    relevant_documents = get_relevant_documents(query=audio_transcript)
    
    context = ""
    for relevant_document in relevant_documents:
        context += f"{relevant_document.page_content}\n"

    llama_instruction = f"""
    <s>
    [INST]
    <<SYS>>
    You are a highly skilled, professional and respectful lawyer.
    Given the text extracts from the Indian Constitution, your job is to provide legally sound and unbiased answers.
    Explain incoherent questions. Refrain from sharing false information.
    You are expected to address hypothetical scenarios with constitutional context.
    Try to answer as briefly as possible.
    Remember! The answer should only pertain to the given context.
    <</SYS>>

    ### BEGIN CONTEXT
    {context}
    ### END CONTEXT

    Question: {audio_transcript}
    [/INST]
    Answer: 
    """

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + CLARIFAI_PAT),)
    user_data_object = resources_pb2.UserAppIDSet(user_id=LLAMA_USER_ID, app_id=LLAMA_APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_data_object,
            model_id=LLAMA_MODEL_ID,
            version_id=LLAMA_MODEL_VERSION_ID,
            inputs=[resources_pb2.Input(data=resources_pb2.Data(text=resources_pb2.Text(raw=llama_instruction)))]
        ),
        metadata=metadata
    )

    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        return post_model_outputs_response
    output_choice = post_model_outputs_response.outputs[0]

    if output_choice is None:
        return None

    output = output_choice.data.text.raw.strip()

    return {"message": output}
