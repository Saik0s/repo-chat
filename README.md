# Repo Chat

Repo chat is a command line utility that allows you to ask questions about a GitHub repository or a local folder.

This is a fork of the [original](https://github.com/mckaywrigley/repo-chat) repository with modifications to make it more user-friendly and easier to use with command line arguments.

## Requirements

In this project we use [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) and [Chroma](https://www.trychroma.com/) as our vector database.

## How To Run

1. Make sure you have set up your OpenAI API key by configuring the `.env` file or setting the `OPENAI_API_KEY` environment variable.

2. Install the required dependencies by running `pip install -r requirements.txt`.

3. Run the `main.py` script with the appropriate command line arguments. Here are the available optional arguments:

   - `--command`: specify whether to embed, load, or query a repository
   - `--path`: specify the path to the repository on your local machine
   - `--url`: specify the URL of the repository on GitHub
   - `--branch`: specify the branch of the repository to use
   - `--collection`: specify the name of the collection to use
   - `--extensions`: specify a comma-separated list of file extensions to include in the embedding process

   For example, to embed a local repository located at `/path/to/repo` and save the resulting embeddings to a collection named `my_collection`, you would run:
   `./main.py --command embed --path /path/to/repo --collection my_collection`
