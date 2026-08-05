from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# SEARCH CONFIGURATION
# ============================================================

CITY = "Charlotte"
STATE = "NC"
RADIUS = 50

# Safety limit to prevent an infinite pagination loop.
MAX_PAGES = 50

PAGE_LOAD_TIMEOUT = 30
WAIT_AFTER_PAGE_LOAD = 3

HEADLESS = False

OUTPUT_FILE = Path("data/hospital_urls.csv")


# ============================================================
# URL GENERATION
# ============================================================

def build_results_url(page_number: int) -> str:
    """
    Build the Medicare Care Compare hospital-results URL
    for a particular page.
    """
    return (
        "https://www.medicare.gov/care-compare/results"
        "?searchType=Hospital"
        f"&page={page_number}"
        f"&city={quote(CITY)}"
        f"&state={quote(STATE)}"
        "&zipcode="
        f"&radius={RADIUS}"
        "&sort=closest"
    )


# ============================================================
# DRIVER SETUP
# ============================================================

def create_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Create and configure the Selenium Chrome driver.
    """
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1500,1000")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")

    options.add_argument(
        "--user-agent=Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    return driver


# ============================================================
# EXTRACTION HELPERS
# ============================================================

def extract_provider_id(url: str) -> str | None:
    """
    Extract the Medicare provider ID from a hospital profile URL.

    Example:
        https://www.medicare.gov/care-compare/details/hospital/450604

    Returns:
        450604
    """
    path_parts = urlparse(url).path.strip("/").split("/")

    if "hospital" not in path_parts:
        return None

    try:
        hospital_index = path_parts.index("hospital")
        provider_id = path_parts[hospital_index + 1].strip()

        return provider_id or None

    except (ValueError, IndexError):
        return None


def dismiss_cookie_banner(driver: webdriver.Chrome) -> None:
    """
    Attempt to close cookie or consent banners.

    The collector continues normally if no banner is present.
    """
    possible_button_texts = [
        "Accept",
        "Accept all",
        "I agree",
        "Close",
        "Continue",
    ]

    for button_text in possible_button_texts:
        xpath = (
            "//button[contains("
            "translate(normalize-space(.), "
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz'), "
            f"'{button_text.lower()}'"
            ")]"
        )

        buttons = driver.find_elements(By.XPATH, xpath)

        for button in buttons:
            try:
                if button.is_displayed() and button.is_enabled():
                    driver.execute_script(
                        "arguments[0].click();",
                        button,
                    )
                    time.sleep(1)
                    return
            except Exception:
                continue


def scroll_results_page(driver: webdriver.Chrome) -> None:
    """
    Scroll through the results page to trigger lazy-loaded links.
    """
    try:
        page_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        positions = [
            0,
            page_height * 0.25,
            page_height * 0.50,
            page_height * 0.75,
            page_height,
        ]

        for position in positions:
            driver.execute_script(
                "window.scrollTo(0, arguments[0]);",
                position,
            )
            time.sleep(0.6)

        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    except Exception:
        pass


def extract_records_from_current_page(
    driver: webdriver.Chrome,
) -> list[dict[str, str]]:
    """
    Extract hospital profile links from the currently opened page.
    """
    links = driver.find_elements(
        By.CSS_SELECTOR,
        'a[href*="/care-compare/details/hospital/"]',
    )

    records: list[dict[str, str]] = []

    for link in links:
        try:
            hospital_name = link.text.strip()
            profile_url = link.get_attribute("href")

            if not hospital_name or not profile_url:
                continue

            provider_id = extract_provider_id(profile_url)

            if not provider_id:
                continue

            records.append(
                {
                    "provider_id": provider_id,
                    "hospital_name": hospital_name,
                    "profile_url": profile_url,
                }
            )

        except Exception:
            continue

    # Remove repeated links from the same page.
    unique_records = {
        record["provider_id"]: record
        for record in records
    }

    return list(unique_records.values())


# ============================================================
# AUTOMATIC PAGINATION
# ============================================================

def collect_hospital_links(
    driver: webdriver.Chrome,
) -> list[dict[str, str]]:
    """
    Visit consecutive Medicare results pages until no new hospitals
    are found.

    Pagination stops when:
    1. No hospital links appear.
    2. A page contains no new provider IDs.
    3. The maximum page limit is reached.
    """
    wait = WebDriverWait(driver, PAGE_LOAD_TIMEOUT)

    collected_records: dict[str, dict[str, str]] = {}
    previous_page_provider_ids: set[str] = set()

    print("=" * 80)
    print("MEDICARE HOSPITAL URL COLLECTION")
    print("=" * 80)
    print(f"City: {CITY}")
    print(f"State: {STATE}")
    print(f"Radius: {RADIUS} miles")
    print(f"Maximum pages: {MAX_PAGES}")
    print()

    for page_number in range(1, MAX_PAGES + 1):
        results_url = build_results_url(page_number)

        print("-" * 80)
        print(f"Opening page {page_number}")
        print(results_url)

        try:
            driver.get(results_url)

        except TimeoutException:
            print(
                f"Page {page_number} exceeded the loading timeout. "
                "Attempting to continue with the available content."
            )

        if page_number == 1:
            dismiss_cookie_banner(driver)

        try:
            wait.until(
                EC.presence_of_all_elements_located(
                    (
                        By.CSS_SELECTOR,
                        'a[href*="/care-compare/details/hospital/"]',
                    )
                )
            )

        except TimeoutException:
            print(
                f"No hospital links appeared on page {page_number}."
            )
            print("Pagination is complete.")
            break

        time.sleep(WAIT_AFTER_PAGE_LOAD)
        scroll_results_page(driver)

        page_records = extract_records_from_current_page(driver)

        if not page_records:
            print(
                f"Page {page_number} contained no hospital records."
            )
            print("Pagination is complete.")
            break

        current_page_provider_ids = {
            record["provider_id"]
            for record in page_records
        }

        # Medicare may redirect an invalid page number to the last
        # available page. This prevents repeatedly processing it.
        if (
            previous_page_provider_ids
            and current_page_provider_ids
            == previous_page_provider_ids
        ):
            print(
                "This page contains the same hospitals as the "
                "previous page."
            )
            print("The final page has been reached.")
            break

        new_records_count = 0

        for record in page_records:
            provider_id = record["provider_id"]

            if provider_id not in collected_records:
                new_records_count += 1

            collected_records[provider_id] = record

        print(
            f"Hospital links found on page: {len(page_records)}"
        )
        print(
            f"New unique hospitals added: {new_records_count}"
        )
        print(
            f"Total unique hospitals collected in this search: "
            f"{len(collected_records)}"
        )

        if new_records_count == 0:
            print(
                "No new provider IDs were found on this page."
            )
            print("Pagination is complete.")
            break

        previous_page_provider_ids = current_page_provider_ids

    else:
        print(
            f"The safety limit of {MAX_PAGES} pages was reached."
        )

    return list(collected_records.values())


# ============================================================
# CSV STORAGE
# ============================================================

def normalize_provider_ids(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize provider IDs so leading/trailing spaces and '.0'
    suffixes do not create false duplicates.
    """
    dataframe = dataframe.copy()

    dataframe["provider_id"] = (
        dataframe["provider_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return dataframe


def save_records(
    records: list[dict[str, str]],
) -> None:
    """
    Append newly collected hospitals to the master URL dataset.

    Existing records are preserved, and duplicate provider IDs
    are removed.
    """
    if not records:
        print("\nNo hospital records were collected.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_dataframe = pd.DataFrame(records)
    new_dataframe = normalize_provider_ids(new_dataframe)

    expected_columns = [
        "provider_id",
        "hospital_name",
        "profile_url",
    ]

    new_dataframe = new_dataframe[expected_columns]

    existing_count = 0

    if OUTPUT_FILE.exists():
        existing_dataframe = pd.read_csv(
            OUTPUT_FILE,
            dtype={"provider_id": str},
        )

        missing_columns = (
            set(expected_columns)
            - set(existing_dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{OUTPUT_FILE} is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        existing_dataframe = existing_dataframe[
            expected_columns
        ]

        existing_dataframe = normalize_provider_ids(
            existing_dataframe
        )

        existing_count = len(existing_dataframe)

        combined_dataframe = pd.concat(
            [
                existing_dataframe,
                new_dataframe,
            ],
            ignore_index=True,
        )

    else:
        combined_dataframe = new_dataframe

    total_before_deduplication = len(combined_dataframe)

    combined_dataframe = (
        combined_dataframe
        .dropna(
            subset=[
                "provider_id",
                "hospital_name",
                "profile_url",
            ]
        )
        .drop_duplicates(
            subset=["provider_id"],
            keep="last",
        )
        .sort_values(
            by=[
                "hospital_name",
                "provider_id",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    duplicates_removed = (
        total_before_deduplication
        - len(combined_dataframe)
    )

    newly_added_to_master = (
        len(combined_dataframe)
        - existing_count
    )

    combined_dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("URL COLLECTION COMPLETED")
    print("=" * 80)

    print(
        f"Hospitals collected in the current search: "
        f"{len(new_dataframe)}"
    )

    print(
        f"Hospitals previously in the master file: "
        f"{existing_count}"
    )

    print(
        f"New hospitals added to the master file: "
        f"{max(newly_added_to_master, 0)}"
    )

    print(
        f"Duplicate records removed: "
        f"{duplicates_removed}"
    )

    print(
        f"Total unique hospitals in the master file: "
        f"{len(combined_dataframe)}"
    )

    print(f"Saved to: {OUTPUT_FILE}")

    print("\nHospitals collected in this search:")

    print(
        new_dataframe[
            [
                "provider_id",
                "hospital_name",
                "profile_url",
            ]
        ].to_string(index=False)
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main() -> None:
    driver = create_driver(headless=HEADLESS)

    try:
        records = collect_hospital_links(driver)
        save_records(records)

        if not HEADLESS:
            input("\nPress Enter to close Chrome...")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()