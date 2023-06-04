import os
import argparse
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import logging
import rich
from rich.traceback import install

install(show_locals=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.format = "%(asctime)s %(levelname)s %(name)s: %(message)s"

load_dotenv()


def generate_vector_store(path, collection_name, allowed_extensions):
    """
    This function generates a vector store from a directory of text files.
    :param path: The path to the directory containing the text files.
    :param collection_name: The name of the collection.
    :param allowed_extensions: A comma separated list of extensions that are allowed.
    """
    # configure these to fit your needs
    exclude_dir = [".git", "node_modules", "public", "assets"]
    exclude_files = ["package-lock.json", ".DS_Store"]
    exclude_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".svg", ".webp", ".mp3", ".wav"]

    documents = []

    for dirpath, dirnames, filenames in os.walk(path):
        # skip directories in exclude_dir
        dirnames[:] = [d for d in dirnames if d not in exclude_dir]

        total = len(filenames)
        current = 0
        logger.info(f"Processing directory {dirpath}, {total} files, path: {dirpath}")
        for file in filenames:
            _, file_extension = os.path.splitext(file)

            current += 1
            logger.info(f"Processing file {current}/{total} {file}")
            # skip files in exclude_files
            if file not in exclude_files and file_extension not in exclude_extensions:
                file_path = os.path.join(dirpath, file)
                if allowed_extensions and not file.endswith(tuple(allowed_extensions)):
                    continue
                loader = TextLoader(file_path, encoding="ISO-8859-1")
                original_documents = loader.load()

                # Split documents according to their extension
                for doc in original_documents:
                    file_extension_clean = file_extension.replace(".", "")
                    if file_extension_clean == "py":
                        file_extension_clean = "python"
                    elif file_extension_clean == "md":
                        file_extension_clean = "markdown"

                    try:
                        text_splitter = RecursiveCharacterTextSplitter.from_language(
                            language=Language(file_extension_clean), chunk_size=500, chunk_overlap=150
                        )
                    except Exception as e:
                        logger.info(
                            f"{e}\nCould not find language for extension {file_extension_clean}, falling back to character splitter"
                        )
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

                    split_documents = text_splitter.split_documents([doc])
                    documents.extend(split_documents)

    logger.info(f"Loaded {len(documents)} documents")

    for doc in documents:
        source = doc.metadata["source"]
        cleaned_source = "/".join(source.split("/")[1:])
        doc.page_content = f"FILE NAME: {cleaned_source}" + "\n###\n" + doc.page_content.replace("\u0000", "")

    embeddings = OpenAIEmbeddings(disallowed_special=set())

    logger.info("Generating vectors")
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        collection_name=collection_name,
        persist_directory="vectors",
    )

    logger.info("Persisting vector store")
    vector_store.persist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script generates a vector store from a directory of text files. Please provide the path to the directory as an argument."
    )
    parser.add_argument("path", type=str, help="path to the directory containing the text files")
    parser.add_argument("collection", type=str, help="name of the collection")
    parser.add_argument("--extensions", type=str, help="comma separated list of extensions that are allowed")
    args = parser.parse_args()
    generate_vector_store(args.path, args.collection, args.extensions.split(",") if args.extensions else [])
