import os
import argparse
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

load_dotenv()

def generate_vector_store(path, collection_name, allowed_extensions):
    """
    This function generates a vector store from a directory of text files.
    :param path: The path to the directory containing the text files.
    :param collection_name: The name of the collection.
    :param allowed_extensions: A comma separated list of extensions that are allowed.
    """
    # configure these to fit your needs
    exclude_dir = ['.git', 'node_modules', 'public']
    exclude_files = ['package-lock.json']

    documents = []

    for dirpath, dirnames, filenames in os.walk(path):
        # skip directories in exclude_dir
        dirnames[:] = [d for d in dirnames if d not in exclude_dir]

        for file in filenames:
            # skip files in exclude_files
            if file not in exclude_files:
                file_path = os.path.join(dirpath, file)
                if allowed_extensions and not file.endswith(
                    tuple(allowed_extensions)
                ):
                    continue
                loader = TextLoader(file_path)
                documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    for doc in docs:
        source = doc.metadata['source']
        cleaned_source = '/'.join(source.split('/')[1:])
        doc.page_content = (
            f"FILE NAME: {cleaned_source}" + "\n###\n" + doc.page_content
        )

    embeddings = OpenAIEmbeddings()

    vector_store = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=collection_name,
        persist_directory="vectors",
    )

    vector_store.persist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script generates a vector store from a directory of text files. Please provide the path to the directory as an argument.')
    parser.add_argument('path', type=str, help='path to the directory containing the text files')
    parser.add_argument('collection', type=str, help='name of the collection')
    parser.add_argument('--extensions', type=str, help='comma separated list of extensions that are allowed')
    args = parser.parse_args()
    generate_vector_store(args.path, args.collection, args.extensions.split(',') if args.extensions else [])
