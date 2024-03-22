import requests
from bs4 import BeautifulSoup
from scraper.logger import get_logger


def get_html(url):
    """Get HTML content from the specified URL."""
    logger = get_logger("RequestHandler")
    logger.info(f"Fetching HTML content from {url}")
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
        return response.content