#!/usr/bin/env python3

import os
import argparse
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings()

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--command", type=str, help="embed, load or query")
parser.add_argument("--path", type=str, default="repo", help="path to the codebase")
parser.add_argument("--url", type=str, help="url of the codebase")
parser.add_argument("--branch", type=str, help="branch of the codebase")
parser.add_argument("--collection", type=str, help="name of the collection")
parser.add_argument("--extensions", type=str, help="comma separated list of extensions that are allowed")
parser.add_argument(
    "--model", type=str, default="gpt-3.5-turbo", help="name of the ChatOpenAI model to use (default: gpt-3.5-turbo)"
)

args = parser.parse_args()

if not args.command:
    args.command = input("\033[34mWhat do you want to do? (embed, load or query)\n\033[0m")

if args.command == "query":
    if not args.collection:
        vector_store = Chroma(persist_directory="vectors")
        collections = vector_store._client.list_collections()
        print("\033[34mAvailable collections:\033[0m")
        for collection in collections:
            print(collection.name)
        args.collection = input("\033[34mWhich collection do you want to query?\n\033[0m")

    vector_store = Chroma(args.collection, embeddings, persist_directory="vectors")

    chat = ChatOpenAI(
        model=args.model,
        max_tokens=500,
        streaming=False,
        temperature=0.5,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )
    while True:
        query = input("\033[34mWhat question do you have about your repo?\n\033[0m")

        if query.lower().strip() == "exit":
            print("\033[31mGoodbye!\n\033[0m")
            break

        print("\n\n")

        from langchain.retrievers import SVMRetriever
        from langchain.embeddings import OpenAIEmbeddings

        all_docs = vector_store._client._get(
            collection_id=vector_store._collection.id, include=["documents", "embeddings", "metadatas"]
        )
        retriever = SVMRetriever(
            embeddings=OpenAIEmbeddings(),
            index=np.array(all_docs["embeddings"]),
            texts=list(all_docs["documents"]),
            k=15,
        )
        from langchain.chains import RetrievalQA

        # matched_docs = retriever.get_relevant_documents(query)
        # matched_docs = vector_store.similarity_search(query, k=6)
        # code_str = "".join(doc.page_content + "\n\n" for doc in matched_docs)
        # print("\n\033[35m" + code_str + "\n\033[32m")

        template = """
        You are Codebase AI. You are a superintelligent AI that answers questions about codebases.

        You are:
        - helpful & friendly
        - good at answering complex questions in simple language
        - an expert in all programming languages
        - able to infer the intent of the user's question

        The user will ask a question about their codebase, and you will answer it.

        When the user asks their question, you will answer it by searching the codebase for the answer.

        Here is the user's question and code file(s) you found to answer the question:

        Question:
        {question}

        Code file(s):
        {context}

        [END OF CODE FILE(S)]

        Now answer the question using the code file(s) above.
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        chain_type_kwargs = {"question_prompt": chat_prompt}
        qa = RetrievalQA.from_chain_type(llm=chat, chain_type="refine", retriever=retriever, verbose=True)
        print("Thinking...")
        result = qa(query)

        # chain = LLMChain(llm=chat, prompt=chat_prompt)

        # result = chain.run(code=code_str, query=query)

        # print("\n\n")
        print(f"Final Result:\n{result}")

        print("\n\n")
elif args.command == "embed":
    if not args.path:
        args.path = input("\033[34mWhat is the path to the codebase?\n\033[0m")
    if not args.collection:
        args.collection = input("\033[34mWhat is the name of the collection?\n\033[0m")
    if not args.extensions:
        args.extensions = input(
            "\033[34mWhat is the comma separated list of extensions that are allowed? (Empty to allow all)\n\033[0m"
        )
    if args.extensions:
        args.extensions = f"--extensions {args.extensions}"
    os.system(f"python embed.py {args.path} {args.collection} {args.extensions}")

elif args.command == "load":
    if not args.url:
        args.url = input("\033[34mWhat is the url of the codebase?\n\033[0m")
    if not args.branch:
        args.branch = input("\033[34mWhat is the branch of the codebase?\n\033[0m")
    os.system(f"python load.py {args.url} {args.branch} --path {args.path}")
