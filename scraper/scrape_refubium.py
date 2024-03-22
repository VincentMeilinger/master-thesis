import requests
from bs4 import BeautifulSoup
from scraper.logger import get_logger
import os

logger = get_logger("RefubiumScraper")

DUMP_DIR = './data/dump/'  # Directory to store scraped data


def parse(html_content) -> dict:
    logger.info("Parsing HTML content")
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find meta tags that have 'name' and 'content' attributes
    meta_tags = soup.find_all('meta', attrs={'name': True, 'content': True})

    filter = ["viewport", "Generator"]

    dataset = {}
    # Extract name and content attributes
    for tag in meta_tags:
        name = tag.get('name')
        if name in filter:  # Filter out unwanted tags
            continue
        content = tag.get('content')
        language = tag.get('xml:lang')
        scheme = tag.get('scheme')
        logger.debug(f"Name: {name}, Content: {content}, Language: {language}, Scheme: {scheme}")

        ds_property = {}
        # If the name is not present, create an empty array
        if name not in dataset:
            dataset[name] = []

        ds_property["content"] = content
        if language:
            ds_property["language"] = language
        if scheme:
            ds_property["scheme"] = scheme

        dataset[name].append(ds_property)

    return dataset


def scrape(url: str, write_to_file: bool = False) -> dict:
    logger.info(f"Scraping {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        logger.error(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        logger.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        logger.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        logger.error(f"Something went wrong: {err}")
    else:
        content = parse(response.content)

        if write_to_file:
            if not os.path.exists(DUMP_DIR):
                os.makedirs(DUMP_DIR)
            file_name = url.strip("https://").replace("/", "_")
            with open(DUMP_DIR + f'{file_name}.json', 'w') as f:
                f.write(str(content))
        return content
