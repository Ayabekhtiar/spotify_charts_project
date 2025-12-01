import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from datetime import datetime, timedelta

# ------------------ CONFIG ------------------
def generate_week_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=7)
    return dates


START_DATE = "2020-06-11"       # Important: must be a Thursday
END_DATE =  "2020-12-31"
DATES = generate_week_dates(START_DATE, END_DATE)

BASE_URL = "https://charts.spotify.com/charts/view/regional-global-weekly/"
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

DOWNLOAD_TIMEOUT = 30
BUTTON_WAIT_TIME = 10
RETRIES = 3

# Random delay settings
def human_delay(a=0.7, b=2.3):
    time.sleep(random.uniform(a, b))

# ------------------------------------------------


def setup_driver():
    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": DOWNLOAD_DIR,
        "download.prompt_for_download": False,
        "safebrowsing.enabled": True,
    }
    options.add_experimental_option("prefs", prefs)

    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()

    try:
        driver.execute_cdp_cmd("Page.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": DOWNLOAD_DIR
        })
    except:
        pass

    return driver


def wait_for_download(before_files):
    end_time = time.time() + DOWNLOAD_TIMEOUT
    while time.time() < end_time:
        after_files = set(os.listdir(DOWNLOAD_DIR))
        new_files = after_files - before_files
        if new_files:
            return list(new_files)[0]
        time.sleep(1)
    return None


def find_download_button(driver):
    for attempt in range(RETRIES):
        try:
            button = WebDriverWait(driver, BUTTON_WAIT_TIME).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-encore-id='buttonTertiary']"))
            )
            return button
        except:
            print(f"  🔁 Retry {attempt+1}/{RETRIES}: scrolling & waiting...")
            driver.execute_script("window.scrollBy(0, 250);")
            human_delay(1.5, 3.2)
    return None


def download_for_date(driver, date):
    url = BASE_URL + date
    print(f"\n🌍 Opening page for {date}...")
    driver.get(url)
    human_delay(1.2, 3.4)              # After page load

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
    human_delay(0.7, 1.8)

    download_button = find_download_button(driver)

    if not download_button:
        print(f"⚠ No download button found for {date}. Skipping...")
        return

    driver.execute_script("arguments[0].style.border='3px solid red'", download_button)
    human_delay(0.8, 2.1)              # Delay before clicking

    before = set(os.listdir(DOWNLOAD_DIR))
    print("⬇ Clicking download icon…")

    try:
        download_button.click()
    except:
        driver.execute_script("arguments[0].click();", download_button)

    downloaded = wait_for_download(before)

    if downloaded:
        print(f"✔ Downloaded: {downloaded}")
    else:
        print(f"❌ Download timed out for {date}")

    human_delay(2.5, 5.1)              # Delay before next iteration


if __name__ == "__main__":
    print(f"📁 Download folder: {DOWNLOAD_DIR}")
    driver = setup_driver()

    print("\n🔐 Opening first page to allow login...")
    driver.get(BASE_URL + DATES[0])
    input("⏸ Please login & accept cookies, then press ENTER once. This step won't repeat. ")

    human_delay(2, 4)

    for date in DATES:
        download_for_date(driver, date)

    print("\n🎉 Finished!")
    print("📦 Files:", os.listdir(DOWNLOAD_DIR))

    driver.quit()
