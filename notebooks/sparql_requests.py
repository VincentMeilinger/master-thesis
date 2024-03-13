import pandas as pd
import requests


def run_query(url, query):
    """Run SPARQL query and return results as a DataFrame."""
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query"
    }
    response = requests.post(url, data=query, headers=headers)

    if response.status_code == 200:
        data = response.json()
        cols = data['head']['vars']
        rows = [{col: row['value'] for col, row in item.items()} for item in data['results']['bindings']]
        return pd.DataFrame(rows, columns=cols)
    else:
        print(f"Error: {response.status_code}")
        return pd.DataFrame()


def fetch_metadata(url, uri):
    """Fetch metadata for the specified URI."""
    query = f"""
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?title ?creator ?date ?description
    WHERE ?resource a dcat:Dataset {{
      <{uri}> dct:title ?title .
      <{uri}> dct:creator ?creator .
      <{uri}> dct:date ?date .
      <{uri}> dct:description ?description .
      <{uri}> a ?type .
      <{uri}> dct:format ?format .
      <{uri}> dct:language ?language .
      <{uri}> dct:subject ?subject .
      <{uri}> dct:identifier ?identifier .
      <{uri}> dct:source ?source .
      <{uri}> dct:publisher ?publisher .
      <{uri}> dct:rights ?rights .
      <{uri}> dct:relation ?relation .
      <{uri}> dct:coverage ?coverage .
      <{uri}> dct:contributor ?contributor .
      FILTER (!isLiteral(?value) || langMatches(lang(?value), "en"))
    }}
    """
    return run_query(url, query)

def download_file(url, path):
    """Download a file from the specified URL to the given path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Error downloading file: {response.status_code}")