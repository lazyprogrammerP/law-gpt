import os

from fastapi import UploadFile

from helpers.documents.loaders import load_txt, load_pdf


def load_document(document: UploadFile):
    document_directory = ".docstore"
    os.makedirs(document_directory, exist_ok=True)

    document_path = f"{document_directory}/{document.filename}"

    if os.path.isfile(document_path):
        os.remove(document_path)

    with open(f".docstore/{document.filename}", "wb+") as document_writer:
        document_writer.write(document.file.read())

    loader = None
    extension = document.filename.split(".")[1]
    if extension == "txt":
        loader = load_txt

    if extension == "pdf":
        loader = load_pdf

    if loader is None:
        raise Exception("Document extension is not supported.")

    return loader(document_path=document_path), document_path
