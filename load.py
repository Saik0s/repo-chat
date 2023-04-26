import argparse
from dotenv import load_dotenv
from langchain.document_loaders import GitLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Load git repo')
    parser.add_argument('url', type=str, help='URL of the git repo')
    parser.add_argument('branch', type=str, help='Branch of the git repo')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    load_dotenv()

    loader = GitLoader(
        clone_url=args.url,
        repo_path='repo',
        branch=args.branch,
    )

    loader.load()

