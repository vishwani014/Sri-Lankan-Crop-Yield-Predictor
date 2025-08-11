import requests
import cdsapi
import pandas as pd
import os
import pdfplumber
import earthaccess
from datetime import datetime
from bs4 import BeautifulSoup
import time
import xarray as xr
from urllib.parse import urljoin
from dotenv import load_dotenv


# current/forecast weather
def fetch_openweather_data(api_key, cities, output_path): 
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    data = []
    for city in cities:
        params = {"q": f"{city},LK", "appid": api_key, "units": "metric"}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data.append({
                "city": city,
                "date": datetime.utcfromtimestamp(json_data["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": json_data["main"]["temp"],
                "rainfall": json_data.get("rain", {}).get("1h", 0),
                "humidity": json_data["main"]["humidity"]
            })
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching data for {city}: {e}")
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df


# historical weather
def fetch_era5_data(years):
    client = cdsapi.Client()
    data = []
    for year in years:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["2m_temperature", "total_precipitation"],
                "year": str(year),
                "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
                "day": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"],
                "time": ["06:00"],
                "area": [9.9, 79.5, 5.9, 81.9],  # Sri Lanka bounding box
                "format": "netcdf",
                "download_format": "unarchived"
            },
            f"data/raw/era5_{year}.nc"
        )
    print(f"ERA5 data saved to era5_{year}.nc")



# Market prices from cbsl(PDF extraction))
def scrape_cbsl_prices(url, output_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    pdf_links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".pdf")]

    data = []
    for link in pdf_links[:10]:
        # pdf_url = f"https://www.cbsl.gov.lk{link}" if link.startswith("/") else link
        pdf_url = urljoin(url, link)
        print(f"Processing: {pdf_url}")
        try:
            pdf_response = requests.get(pdf_url, headers=headers)
            pdf_response.raise_for_status()

            with open("temp.pdf", "wb") as f:
                f.write(pdf_response.content)

            with pdfplumber.open("temp.pdf") as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if table:
                        for row in table[1:]:
                            if row and len(row) >= 3 and row[0] and row[2]:
                                data.append({
                                    "crop": row[0].strip(),
                                    "date": row[1].strip(),
                                    "price": float(row[2].replace("LKR", "").strip())
                                })
            time.sleep(2)
        except Exception as e:
            print(f"Error processing {pdf_url}: {e}")
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    return df


# NASA Earthdata: NDVI
def fetch_ndvi_data(start_date, end_date, output_path):
    auth = earthaccess.login()
    results = earthaccess.search_data(
        short_name="MOD13Q1",
        bounding_box=(79.5, 5.5, 81.5, 9.5),  # Sri Lanka
        temporal=(start_date, end_date)
    )
    data = []
    processed_dates = set()
    for result in results[:10]:  # Limit for free tier
        # Process NetCDF files (simplified, use xarray)
        data.append({
            "date": result["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"],
            "ndvi": 0.5  # Placeholder, extract from NetCDF
        })

    # Create DataFrame and save to CSV
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values(by="date")  # Sort by date
        df.to_csv(output_path, index=False)
        print(f"NDVI data saved to {output_path}")
        return df
    else:
        print("No NDVI data retrieved.")
        return pd.DataFrame()
    
    # df = pd.DataFrame(data)
    # df.to_csv(output_path, index=False)
    # return df


if __name__ == "__main__":
    load_dotenv()

    os.makedirs("data/raw", exist_ok=True)

    # set api key for open  weather
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not set in .env file")
    
    CITIES = ["Kurunegala", "Anuradhapura", "Kandy", "Matale"]
    fetch_openweather_data(API_KEY, CITIES, "data/raw/weather_current.csv")

    YEARS = [2023, 2024]
    fetch_era5_data(YEARS)

    CBSL_URL = "https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report"
    scrape_cbsl_prices(CBSL_URL, "data/raw/market_prices.csv")

    fetch_ndvi_data("2023-01-01", "2023-12-31", "data/raw/ndvi.csv")
