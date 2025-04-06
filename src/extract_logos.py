import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from PIL import Image
from io import BytesIO
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from src.load_data import load_dataset

HEADERS = {'User-Agent': 'Mozilla/5.0'}
OUTPUT_DIR = "logos"
MAX_WORKERS = 30
TIMEOUT = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_image(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()  # Ensure the request was successful
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"[✖] Error downloading image from {url}: {e}")
        return None

# Function to extract logo from standard <img> tags
def extract_logo_url(soup, base_url):
    for img in soup.find_all("img"):
        attrs = " ".join([str(img.get(attr, "")).lower() for attr in ["alt", "src", "class", "id"]])
        if "logo" in attrs:
            src = img.get("src")
            if src:
                # Ensure src is joined properly with base_url
                logo_url = urljoin(base_url, src)
                return logo_url
    return None

def extract_background_logo_url(soup, base_url):
    # Look for divs, spans, or any block-level element that may contain a background image
    for tag in soup.find_all(["div", "span", "header", "section"]):
        style = tag.get("style", "")
        if "background-image" in style:
            match = re.search(r'url\(["\']?(.*?)["\']?\)', style)
            if match:
                return urljoin(base_url, match.group(1))
    return None

# Function to extract favicon from <link> tags
def extract_favicon_url(soup, base_url):
    for link in soup.find_all("link", rel=lambda x: x and 'icon' in x):
        href = link.get("href")
        if href:
            return urljoin(base_url, href)
    return urljoin(base_url, "/favicon.ico")

# Function to save an image locally
def save_image(image, filename):
    try:
        image.convert("RGB").save(filename)
        print(f"[✔] Saved {filename}")
    except Exception as e:
        print(f"[✖] Could not save {filename}: {e}")

# Main processing function to handle domains and logo extraction
def process_domain(domain):
    domain = domain.strip().lower()
    filename = os.path.join(OUTPUT_DIR, f"{domain.replace('.', '_')}.png")

    if os.path.exists(filename):
        print(f"[!] Already exists: {domain}")
        return

    try:
        # Ensure URL is properly formed with http/https
        parsed_url = urlparse(domain)
        if not parsed_url.scheme:
            base_url = f"http://{domain}"
        else:
            base_url = domain

        response = requests.get(base_url, headers=HEADERS, timeout=TIMEOUT)
        soup = BeautifulSoup(response.text, "html.parser")

        # Try extracting logo using standard <img> tags
        logo_url = extract_logo_url(soup, base_url)
        if logo_url:
            img = download_image(logo_url)
            if img:
                save_image(img, filename)
                return

        # Try extracting background logo
        background_logo_url = extract_background_logo_url(soup, base_url)
        if background_logo_url:
            img = download_image(background_logo_url)
            if img:
                save_image(img, filename)
                return

        # Fallback to favicon if no logo is found
        favicon_url = extract_favicon_url(soup, base_url)
        if favicon_url:
            img = download_image(favicon_url)
            if img:
                save_image(img, filename)
                return

        print(f"[!] No logo found for {domain}")

    except Exception as e:
        print(f"[✖] Error processing {domain}: {e}")

# Function to process domains in parallel
def extract_logos_parallel():
    df = load_dataset()
    domains = df["domain"].dropna().unique()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_domain, domain) for domain in domains]
        for future in as_completed(futures):
            future.result()  

if __name__ == "__main__":
    extract_logos_parallel()
