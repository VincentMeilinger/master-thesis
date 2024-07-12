import argparse
import json
import os

def search_who_is_who(search_pub_id: str):
    file_path = os.path.join('./data/datasets/IND-WhoIsWho', 'pid_to_info_all.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    with open(file_path, 'r') as file:
        data = json.load(file)

    if not data:
        raise ValueError(f"Unable to parse {file_path}.")

    for pub_id, values in data.items():
        if pub_id == search_pub_id:
            # Pretty print the publication data
            print("Publication found:")
            print(json.dumps(values, indent=4))
            break

parser = argparse.ArgumentParser(description="Search the publication database files for a specific publication by ID.")
parser.add_argument(
    '--pub_id', '-id',
    type=str,
    help='The publication ID to search for.'
)
args = parser.parse_args()

if args.pub_id:
    search_who_is_who(args.pub_id)
