import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
import pandas as pd
from load_data import load_dataset


from load_data import load_dataset

def extract_logo_url(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for img in soup.find_all("img"):
        #For every <img src = "logo..", alt ="logo.."> I create a vector and look for the keyword "logo"
        attrs = " ".join([str(img.get(attr, "")).lower() for attr in ["alt", "src", "class", "id"]]) #Combine all results into one string, e.g main logo /logo.png header-img
        if "logo" in attrs:
            return urljoin(base_url, img.get("src"))
    return None

def save_logo(img_url, domain, output_dir="logos"):
    #Check if the path is already in logos directory
    path = os.path.join(output_dir, f"{domain.replace('.','_')}.png")
    if os.path.exists(path):
        print(f"[!] Logo for {domain} already exists. Skipping...")
        return
    try:
        #Convert raw image to file-like object and save at path
        response = requests.get(img_url, timeout=10)
        image = Image.open(BytesIO(response.content))
        image.save(path)
        print(f"[✔] Saved logo for {domain}")
    except Exception as e:
        print(f"[✖] Failed to save logo for {domain}: {e}")

def extract_favicon_url(html, base_url):
    #Backup method, looking for favicons if we dont find any img "logo.."
    soup = BeautifulSoup(html, "html.parser")
    favicon_url = None

    for link in soup.find_all("link", rel=["icon", "shortcut icon", "apple-touch-icon"]):
            favicon_url = link.get("href")
            if favicon_url:
                break
    if favicon_url:
        return urljoin(base_url, favicon_url)
    return None
def save_favicon(favicon_url,domain,output_dir = "logos"):
    path = os.path.join(output_dir, f"{domain.replace('.','_')}_favicon.png")
    if os.path.exists(path):
        print(f"[!] Logo for {domain} already exists. Skipping...")
        return

    try:
        response = requests.get(favicon_url, timeout=10)
        image = Image.open(BytesIO(response.content))
        path = os.path.join(output_dir, f"{domain.replace('.','_')}_favicon.png")
        image.save(path)
    except Exception as e:
        print(f"[✖] Failed to save favicon for {domain}: {e}")

def extract_logos():
    os.makedirs("logos", exist_ok=True)
    df = load_dataset()
    #For every domain in our dataset
    for domain in df["domain"]:
        try:
            #We load domain url and raw html
            url = f"http://{domain}"
            html = requests.get(url, timeout=10).text
            logo_url = extract_logo_url(html, url)
            if logo_url:
                save_logo(logo_url, domain)
            else:
                favicon_url = extract_favicon_url(html, url)
                if favicon_url:
                    save_favicon(favicon_url, domain)
                else:
                    print(f"[!] No logo or favicon found for {domain}")


        except Exception as e:
            print(f"[✖] Error fetching {domain}: {e}")

if __name__ == "__main__":
    extract_logos()
