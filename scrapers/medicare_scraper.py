from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/hospital_urls.csv")
OUTPUT_FILE = Path("data/medicare_raw.csv")
FAILED_FILE = Path("data/medicare_failed.csv")

DEBUG_DIRECTORY = Path("data/debug_profiles")
SCREENSHOT_DIRECTORY = DEBUG_DIRECTORY / "screenshots"
HTML_DIRECTORY = DEBUG_DIRECTORY / "html"
TEXT_DIRECTORY = DEBUG_DIRECTORY / "text"

HEADLESS = True

PAGE_TIMEOUT_SECONDS = 40
ELEMENT_TIMEOUT_SECONDS = 20
WAIT_AFTER_PAGE_LOAD = 2
WAIT_AFTER_CLICK = 1
WAIT_BETWEEN_HOSPITALS = 1

# Set to an integer such as 2 or 10 while testing.
# Use None to scrape every hospital in hospital_urls.csv.
SCRAPE_LIMIT: int | None = None


# ============================================================
# DRIVER SETUP
# ============================================================

def create_driver(headless: bool = False) -> webdriver.Chrome:
    """
    Create and configure a Selenium Chrome driver.

    Modern Selenium uses Selenium Manager, so a separate
    webdriver-manager package is not required.
    """
    options = Options()

    if headless:
        options.add_argument("--headless=new")

    options.add_argument("--window-size=1600,1100")
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

    driver.set_page_load_timeout(PAGE_TIMEOUT_SECONDS)

    return driver


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(value: Any) -> str | None:
    """
    Normalize whitespace and return None for empty values.
    """
    if value is None:
        return None

    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None


def parse_integer(value: str | None) -> int | None:
    """
    Extract the first integer from a string.

    Example:
        '320 clinicians affiliated' -> 320
    """
    if not value:
        return None

    match = re.search(r"\d[\d,]*", value)

    if not match:
        return None

    return int(match.group(0).replace(",", ""))


def parse_rating(value: str | None) -> float | None:
    """
    Convert an ARIA label such as '4 stars' or '4.5 stars'
    into a numeric value.
    """
    if not value:
        return None

    match = re.search(r"([0-5](?:\.\d+)?)", value)

    if not match:
        return None

    return float(match.group(1))


def split_address(
    address_lines: list[str],
) -> dict[str, str | None]:
    """
    Convert Medicare address lines into street, city, state,
    and ZIP fields.

    Expected form:
        1020 South State Highway 16
        Fredericksburg, TX 78624
    """
    result = {
        "address": None,
        "city": None,
        "state": None,
        "zip_code": None,
    }

    cleaned_lines = [
        clean_text(line)
        for line in address_lines
        if clean_text(line)
    ]

    if not cleaned_lines:
        return result

    if len(cleaned_lines) >= 2:
        result["address"] = cleaned_lines[0]
        city_state_zip = cleaned_lines[1]
    else:
        city_state_zip = cleaned_lines[0]

    match = re.search(
        r"^(.*?),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$",
        city_state_zip,
    )

    if match:
        result["city"] = clean_text(match.group(1))
        result["state"] = match.group(2)
        result["zip_code"] = match.group(3)
    elif len(cleaned_lines) == 1:
        result["address"] = cleaned_lines[0]

    return result


# ============================================================
# SELENIUM HELPERS
# ============================================================

def safe_find_elements(
    driver: webdriver.Chrome,
    by: str,
    selector: str,
) -> list[WebElement]:
    """
    Find elements without raising an exception.
    """
    try:
        return driver.find_elements(by, selector)
    except Exception:
        return []


def first_non_empty_text(
    elements: list[WebElement],
) -> str | None:
    """
    Return the first non-empty visible text from a list.
    """
    for element in elements:
        try:
            text = clean_text(element.text)

            if text:
                return text
        except StaleElementReferenceException:
            continue

    return None


def scroll_page(driver: webdriver.Chrome) -> None:
    """
    Scroll through the complete page to trigger lazy loading.
    """
    try:
        page_height = driver.execute_script(
            "return document.body.scrollHeight"
        )

        positions = [
            0,
            page_height * 0.20,
            page_height * 0.40,
            page_height * 0.60,
            page_height * 0.80,
            page_height,
        ]

        for position in positions:
            driver.execute_script(
                "window.scrollTo(0, arguments[0]);",
                position,
            )
            time.sleep(0.6)

        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)

    except Exception:
        pass


def click_element(
    driver: webdriver.Chrome,
    element: WebElement,
) -> bool:
    """
    Click an element normally, then fall back to JavaScript.
    """
    try:
        driver.execute_script(
            "arguments[0].scrollIntoView("
            "{block: 'center'});",
            element,
        )
        time.sleep(0.5)

        element.click()
        return True

    except (
        ElementClickInterceptedException,
        StaleElementReferenceException,
    ):
        try:
            driver.execute_script(
                "arguments[0].click();",
                element,
            )
            return True
        except Exception:
            return False

    except Exception:
        return False


def click_by_visible_text(
    driver: webdriver.Chrome,
    texts: list[str],
) -> bool:
    """
    Find and click a button, tab, or link using visible text.
    Matching is case-insensitive.
    """
    for text in texts:
        normalized = text.lower()

        xpath = (
            "//*[self::button or self::a or @role='tab']"
            "[contains("
            "translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            "'abcdefghijklmnopqrstuvwxyz'),"
            f"'{normalized}'"
            ")]"
        )

        elements = safe_find_elements(
            driver,
            By.XPATH,
            xpath,
        )

        for element in elements:
            try:
                if element.is_displayed() and element.is_enabled():
                    if click_element(driver, element):
                        time.sleep(WAIT_AFTER_CLICK)
                        return True
            except StaleElementReferenceException:
                continue

    return False


# ============================================================
# PAGE SECTION EXTRACTION
# ============================================================

def get_body_text(
    driver: webdriver.Chrome,
) -> str:
    """
    Return all visible text from the page.
    """
    body = driver.find_element(By.TAG_NAME, "body")
    return body.text.strip()


def extract_text_after_label(
    body_text: str,
    label: str,
    stop_labels: list[str],
) -> str | None:
    """
    Extract text appearing after a label until the next known label.

    This is used as a fallback when the page DOM does not provide
    a stable selector.
    """
    escaped_stops = "|".join(
        re.escape(stop)
        for stop in stop_labels
    )

    pattern = (
        rf"{re.escape(label)}\s*"
        rf"(.+?)"
        rf"(?=\n(?:{escaped_stops})\b|$)"
    )

    match = re.search(
        pattern,
        body_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    return clean_text(match.group(1))


def extract_hospital_name(
    driver: webdriver.Chrome,
    expected_name: str,
) -> str:
    """
    Extract the profile hospital name.
    """
    h1_elements = safe_find_elements(
        driver,
        By.TAG_NAME,
        "h1",
    )

    name = first_non_empty_text(h1_elements)

    if name and name.lower() not in {
        "medicare",
        "care compare",
    }:
        return name

    # Medicare sometimes uses non-h1 headings.
    xpath = (
        "//*[contains("
        "translate(normalize-space(.),"
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
        "'abcdefghijklmnopqrstuvwxyz'),"
        f"'{expected_name.lower()}'"
        ")]"
    )

    matches = safe_find_elements(
        driver,
        By.XPATH,
        xpath,
    )

    found = first_non_empty_text(matches)

    return found or expected_name


def extract_location(
    driver: webdriver.Chrome,
    body_text: str,
) -> dict[str, str | None]:
    """
    Extract street address, city, state, and ZIP code.
    """
    result = {
        "address": None,
        "city": None,
        "state": None,
        "zip_code": None,
    }

    # Look for address patterns directly in visible text.
    address_match = re.search(
        r"LOCATION\s*\n"
        r"(.+?)\n"
        r"([^,\n]+),\s*([A-Z]{2})\s+"
        r"(\d{5}(?:-\d{4})?)",
        body_text,
        flags=re.IGNORECASE,
    )

    if address_match:
        result["address"] = clean_text(
            address_match.group(1)
        )
        result["city"] = clean_text(
            address_match.group(2)
        )
        result["state"] = address_match.group(3).upper()
        result["zip_code"] = address_match.group(4)

        return result

    # Generic US address fallback.
    generic_match = re.search(
        r"(\d+\s+[^\n]+)\n"
        r"([^,\n]+),\s*([A-Z]{2})\s+"
        r"(\d{5}(?:-\d{4})?)",
        body_text,
    )

    if generic_match:
        result["address"] = clean_text(
            generic_match.group(1)
        )
        result["city"] = clean_text(
            generic_match.group(2)
        )
        result["state"] = generic_match.group(3)
        result["zip_code"] = generic_match.group(4)

    return result


def extract_phone_number(
    body_text: str,
) -> str | None:
    """
    Extract a standard US telephone number.
    """
    match = re.search(
        r"\(\d{3}\)\s*\d{3}-\d{4}",
        body_text,
    )

    return match.group(0) if match else None


def extract_hospital_type(
    body_text: str,
) -> str | None:
    """
    Extract the hospital type from the Details section.
    """
    patterns = [
        (
            r"Hospital type\s*\n"
            r"([^\n]+)"
        ),
        (
            r"Hospital type\s*[:\-]?\s*"
            r"([^\n]+)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            body_text,
            flags=re.IGNORECASE,
        )

        if match:
            return clean_text(match.group(1))

    return None


def extract_emergency_services(
    body_text: str,
) -> str | None:
    """
    Extract Yes/No emergency-service availability.
    """
    patterns = [
        (
            r"Provides emergency services\?\s*\n"
            r"(Yes|No)"
        ),
        (
            r"Provides emergency services\?"
            r"\s*[:\-]?\s*(Yes|No)"
        ),
        (
            r"Emergency services"
            r"\s*[:\-]?\s*(Yes|No)"
        ),
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            body_text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).capitalize()

    return None


def extract_affiliated_clinicians(
    body_text: str,
) -> int | None:
    """
    Extract the number of affiliated clinicians.
    """
    patterns = [
        r"([\d,]+)\s+clinicians affiliated",
        r"Affiliated Doctors\s*&\s*Clinicians.*?([\d,]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            body_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            return int(
                match.group(1).replace(",", "")
            )

    return None


def extract_data_last_updated(
    body_text: str,
) -> str | None:
    """
    Extract Medicare's displayed last-updated date.
    """
    match = re.search(
        r"Data last updated:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        body_text,
        flags=re.IGNORECASE,
    )

    return clean_text(match.group(1)) if match else None


# ============================================================
# RATING EXTRACTION
# ============================================================

def rating_element_near_label(
    driver: webdriver.Chrome,
    labels: list[str],
) -> WebElement | None:
    """
    Find the first rating span after a specified label.

    Medicare stores the rating in:
        <span role="img" aria-label="4 stars">
    """
    for label in labels:
        label_lower = label.lower()

        xpath_options = [
            (
                "//*[contains("
                "translate(normalize-space(.),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),"
                f"'{label_lower}'"
                ")]"
                "/following::span"
                "[@role='img' and contains(@aria-label, 'star')]"
                "[1]"
            ),
            (
                "//*[contains("
                "translate(normalize-space(.),"
                "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                "'abcdefghijklmnopqrstuvwxyz'),"
                f"'{label_lower}'"
                ")]"
                "/ancestor::*[self::div or self::section][1]"
                "//span[@role='img' and "
                "contains(@aria-label, 'star')]"
                "[1]"
            ),
        ]

        for xpath in xpath_options:
            elements = safe_find_elements(
                driver,
                By.XPATH,
                xpath,
            )

            for element in elements:
                aria_label = element.get_attribute(
                    "aria-label"
                )

                if aria_label and "star" in aria_label.lower():
                    return element

    return None


def extract_rating_near_label(
    driver: webdriver.Chrome,
    labels: list[str],
) -> dict[str, Any]:
    """
    Extract both raw and numeric forms of a rating.
    """
    element = rating_element_near_label(
        driver,
        labels,
    )

    if element is None:
        return {
            "raw": None,
            "numeric": None,
        }

    raw_rating = clean_text(
        element.get_attribute("aria-label")
    )

    return {
        "raw": raw_rating,
        "numeric": parse_rating(raw_rating),
    }


def extract_all_ratings(
    driver: webdriver.Chrome,
) -> list[str]:
    """
    Extract all star-rating ARIA labels for debugging.
    """
    elements = safe_find_elements(
        driver,
        By.CSS_SELECTOR,
        'span[role="img"][aria-label*="star"]',
    )

    ratings: list[str] = []

    for element in elements:
        raw = clean_text(
            element.get_attribute("aria-label")
        )

        if raw:
            ratings.append(raw)

    return ratings


# ============================================================
# QUALITY SECTION EXTRACTION
# ============================================================

QUALITY_CATEGORIES = [
    "Timely & effective care",
    "Complications & deaths",
    "Unplanned hospital visits",
    "Maternal health",
    "Patient-reported outcomes",
    "Psychiatric unit services",
    "Payment",
]


def extract_available_quality_categories(
    body_text: str,
) -> list[str]:
    """
    Return the quality categories visible on the page.
    """
    visible_categories: list[str] = []

    for category in QUALITY_CATEGORIES:
        if category.lower() in body_text.lower():
            visible_categories.append(category)

    return visible_categories


def extract_quality_section_text(
    body_text: str,
) -> str | None:
    """
    Extract the visible Quality section text.

    The exact measure values may require opening category controls.
    The original section text is preserved for later parsing.
    """
    start_patterns = [
        r"\nQUALITY\s*\n",
        r"Choose a category to see how this hospital scores",
    ]

    start_position = None

    for pattern in start_patterns:
        match = re.search(
            pattern,
            body_text,
            flags=re.IGNORECASE,
        )

        if match:
            start_position = match.start()
            break

    if start_position is None:
        return None

    remaining_text = body_text[start_position:]

    stop_patterns = [
        r"\nDETAILS\s*\n",
        r"\nAFFILIATED DOCTORS",
        r"\nLOCATION\s*\n",
    ]

    end_position = len(remaining_text)

    for pattern in stop_patterns:
        match = re.search(
            pattern,
            remaining_text,
            flags=re.IGNORECASE,
        )

        if match and match.start() > 0:
            end_position = min(
                end_position,
                match.start(),
            )

    return clean_text(
        remaining_text[:end_position]
    )


def click_quality_tab(
    driver: webdriver.Chrome,
) -> bool:
    """
    Open the Quality tab if a clickable Quality tab exists.
    """
    xpath_options = [
        (
            "//*[@role='tab' and "
            "contains("
            "translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            "'abcdefghijklmnopqrstuvwxyz'),"
            "'quality'"
            ")]"
        ),
        (
            "//button[normalize-space()='Quality']"
        ),
        (
            "//a[normalize-space()='Quality']"
        ),
    ]

    for xpath in xpath_options:
        elements = safe_find_elements(
            driver,
            By.XPATH,
            xpath,
        )

        for element in elements:
            try:
                if element.is_displayed():
                    if click_element(driver, element):
                        time.sleep(WAIT_AFTER_CLICK)
                        return True
            except Exception:
                continue

    return False


def extract_category_panel_text(
    driver: webdriver.Chrome,
    category_name: str,
) -> str | None:
    """
    Click a quality category and capture newly visible text.

    Because the exact Medicare category DOM can differ, this
    function uses visible text and a page snapshot rather than
    relying on unstable generated Angular class names.
    """
    category_lower = category_name.lower()

    xpath_options = [
        (
            "//*[@role='tab' or self::button or self::a]"
            "[contains("
            "translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            "'abcdefghijklmnopqrstuvwxyz'),"
            f"'{category_lower}'"
            ")]"
        ),
        (
            "//*[self::button or self::a]"
            "[contains("
            "translate(normalize-space(.),"
            "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
            "'abcdefghijklmnopqrstuvwxyz'),"
            f"'{category_lower}'"
            ")]"
        ),
    ]

    clicked = False

    for xpath in xpath_options:
        elements = safe_find_elements(
            driver,
            By.XPATH,
            xpath,
        )

        for element in elements:
            try:
                if element.is_displayed() and element.is_enabled():
                    if click_element(driver, element):
                        clicked = True
                        time.sleep(WAIT_AFTER_CLICK)
                        break
            except Exception:
                continue

        if clicked:
            break

    if not clicked:
        return None

    scroll_page(driver)

    body_text = get_body_text(driver)

    # Preserve the portion beginning at the selected category.
    category_position = body_text.lower().find(
        category_lower
    )

    if category_position == -1:
        return None

    section = body_text[category_position:]

    stop_positions = []

    for other_category in QUALITY_CATEGORIES:
        if other_category.lower() == category_lower:
            continue

        position = section.lower().find(
            other_category.lower(),
            len(category_name),
        )

        if position > 0:
            stop_positions.append(position)

    for stop_label in [
        "\nDETAILS\n",
        "\nAFFILIATED DOCTORS",
        "\nLOCATION\n",
    ]:
        position = section.upper().find(
            stop_label.upper()
        )

        if position > 0:
            stop_positions.append(position)

    if stop_positions:
        section = section[:min(stop_positions)]

    return clean_text(section)


def extract_quality_category_texts(
    driver: webdriver.Chrome,
    available_categories: list[str],
) -> dict[str, str | None]:
    """
    Attempt to open each quality category and capture its text.
    """
    results: dict[str, str | None] = {}

    click_quality_tab(driver)

    for category in available_categories:
        print(f"    Inspecting quality category: {category}")

        text = extract_category_panel_text(
            driver,
            category,
        )

        column_name = (
            category.lower()
            .replace("&", "and")
            .replace("-", "_")
            .replace(" ", "_")
        )

        results[
            f"{column_name}_text"
        ] = text

    return results


# ============================================================
# DEBUG OUTPUT
# ============================================================

def save_debug_files(
    driver: webdriver.Chrome,
    provider_id: str,
    body_text: str,
) -> None:
    """
    Save screenshot, HTML, and visible text for each hospital.
    """
    SCREENSHOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    HTML_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    TEXT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    screenshot_path = (
        SCREENSHOT_DIRECTORY
        / f"{provider_id}.png"
    )

    html_path = (
        HTML_DIRECTORY
        / f"{provider_id}.html"
    )

    text_path = (
        TEXT_DIRECTORY
        / f"{provider_id}.txt"
    )

    driver.save_screenshot(
        str(screenshot_path)
    )

    html_path.write_text(
        driver.page_source,
        encoding="utf-8",
    )

    text_path.write_text(
        body_text,
        encoding="utf-8",
    )


# ============================================================
# MAIN PROFILE SCRAPER
# ============================================================

def scrape_hospital_profile(
    driver: webdriver.Chrome,
    provider_id: str,
    expected_name: str,
    profile_url: str,
) -> dict[str, Any]:
    """
    Scrape one Medicare Care Compare hospital profile.
    """
    wait = WebDriverWait(
        driver,
        ELEMENT_TIMEOUT_SECONDS,
    )

    driver.get(profile_url)

    wait.until(
        EC.presence_of_element_located(
            (By.TAG_NAME, "body")
        )
    )

    # Wait for the hospital profile content.
    try:
        wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//*[contains("
                    "translate(normalize-space(.),"
                    "'ABCDEFGHIJKLMNOPQRSTUVWXYZ',"
                    "'abcdefghijklmnopqrstuvwxyz'),"
                    "'hospital type'"
                    ")]",
                )
            )
        )
    except TimeoutException:
        # Continue because some fields may still be available.
        pass

    time.sleep(WAIT_AFTER_PAGE_LOAD)
    scroll_page(driver)

    body_text = get_body_text(driver)

    hospital_name = extract_hospital_name(
        driver,
        expected_name,
    )

    location = extract_location(
        driver,
        body_text,
    )

    overall_rating = extract_rating_near_label(
        driver,
        [
            "Overall star rating",
        ],
    )

    patient_survey_rating = extract_rating_near_label(
        driver,
        [
            "Inpatient survey rating",
            "Patient survey rating",
            "Inpatient survey",
        ],
    )

    all_rating_labels = extract_all_ratings(driver)

    hospital_type = extract_hospital_type(
        body_text
    )

    emergency_services = extract_emergency_services(
        body_text
    )

    affiliated_clinicians = extract_affiliated_clinicians(
        body_text
    )

    data_last_updated = extract_data_last_updated(
        body_text
    )

    available_quality_categories = (
        extract_available_quality_categories(
            body_text
        )
    )

    visible_quality_text = extract_quality_section_text(
        body_text
    )

    quality_category_texts = (
        extract_quality_category_texts(
            driver,
            available_quality_categories,
        )
    )

    # Re-read after quality interactions for debugging.
    final_body_text = get_body_text(driver)

    save_debug_files(
        driver,
        provider_id,
        final_body_text,
    )

    record: dict[str, Any] = {
        "provider_id": provider_id,
        "hospital_name": hospital_name,
        "address": location["address"],
        "city": location["city"],
        "state": location["state"],
        "zip_code": location["zip_code"],
        "phone_number": extract_phone_number(
            body_text
        ),
        "hospital_type": hospital_type,
        "emergency_services": emergency_services,
        "overall_rating": overall_rating["numeric"],
        "overall_rating_raw": overall_rating["raw"],
        "patient_survey_rating": (
            patient_survey_rating["numeric"]
        ),
        "patient_survey_rating_raw": (
            patient_survey_rating["raw"]
        ),
        "affiliated_clinicians": affiliated_clinicians,
        "quality_categories": " | ".join(
            available_quality_categories
        ),
        "quality_section_text": visible_quality_text,
        "all_star_rating_labels": " | ".join(
            all_rating_labels
        ),
        "data_last_updated": data_last_updated,
        "profile_url": profile_url,
        "scraped_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "scraping_status": "success",
        "scraping_error": None,
    }

    record.update(quality_category_texts)

    return record


# ============================================================
# DATASET EXECUTION
# ============================================================

def validate_input_file() -> pd.DataFrame:
    """
    Load and validate hospital_urls.csv.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_FILE}\n"
            "Run medicare_url_collector.py first."
        )

    hospitals = pd.read_csv(
        INPUT_FILE,
        dtype={
            "provider_id": str,
            "hospital_name": str,
            "profile_url": str,
        },
    )

    required_columns = {
        "provider_id",
        "hospital_name",
        "profile_url",
    }

    missing_columns = (
        required_columns
        - set(hospitals.columns)
    )

    if missing_columns:
        raise ValueError(
            "hospital_urls.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    hospitals = hospitals.dropna(
        subset=[
            "provider_id",
            "hospital_name",
            "profile_url",
        ]
    )

    hospitals["provider_id"] = (
        hospitals["provider_id"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    hospitals["hospital_name"] = (
        hospitals["hospital_name"]
        .astype(str)
        .str.strip()
    )

    hospitals["profile_url"] = (
        hospitals["profile_url"]
        .astype(str)
        .str.strip()
    )

    hospitals = hospitals.drop_duplicates(
        subset=["provider_id"]
    )

    if SCRAPE_LIMIT is not None:
        hospitals = hospitals.head(
            SCRAPE_LIMIT
        )

    return hospitals.reset_index(drop=True)

def normalize_provider_id_series(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize Medicare provider IDs while preserving leading zeros.
    """
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def load_existing_successful_data() -> pd.DataFrame:
    """
    Load hospitals already saved in medicare_raw.csv.
    """
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        OUTPUT_FILE,
        dtype={"provider_id": str},
    )

    if "provider_id" not in existing.columns:
        raise ValueError(
            f"{OUTPUT_FILE} does not contain a provider_id column."
        )

    existing["provider_id"] = normalize_provider_id_series(
        existing["provider_id"]
    )

    return existing


def get_existing_provider_ids() -> set[str]:
    """
    Return provider IDs already successfully scraped.
    """
    existing = load_existing_successful_data()

    if existing.empty:
        return set()

    return set(
        existing["provider_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )


def filter_unscraped_hospitals(
    hospitals: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Remove hospitals already present in medicare_raw.csv.
    """
    existing_provider_ids = get_existing_provider_ids()

    remaining = hospitals[
        ~hospitals["provider_id"].isin(existing_provider_ids)
    ].copy()

    skipped_count = len(hospitals) - len(remaining)

    return remaining.reset_index(drop=True), skipped_count


def save_progress(
    successful_records: list[dict[str, Any]],
    failed_records: list[dict[str, Any]],
) -> None:
    """
    Preserve previous records, append current records, and remove
    duplicate provider IDs.

    The function is called after every hospital so progress is not
    lost when the scraper is interrupted.
    """
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save successful records
    # --------------------------------------------------------

    if successful_records:
        current_successful = pd.DataFrame(
            successful_records
        )

        current_successful["provider_id"] = (
            normalize_provider_id_series(
                current_successful["provider_id"]
            )
        )

        if OUTPUT_FILE.exists():
            existing_successful = pd.read_csv(
                OUTPUT_FILE,
                dtype={"provider_id": str},
            )

            if not existing_successful.empty:
                existing_successful["provider_id"] = (
                    normalize_provider_id_series(
                        existing_successful["provider_id"]
                    )
                )

            combined_successful = pd.concat(
                [
                    existing_successful,
                    current_successful,
                ],
                ignore_index=True,
                sort=False,
            )
        else:
            combined_successful = current_successful

        combined_successful = (
            combined_successful
            .dropna(subset=["provider_id"])
            .drop_duplicates(
                subset=["provider_id"],
                keep="last",
            )
            .sort_values(
                by=["provider_id"],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        combined_successful.to_csv(
            OUTPUT_FILE,
            index=False,
        )

    # --------------------------------------------------------
    # Save failed records
    # --------------------------------------------------------

    if failed_records:
        current_failed = pd.DataFrame(
            failed_records
        )

        current_failed["provider_id"] = (
            normalize_provider_id_series(
                current_failed["provider_id"]
            )
        )

        if FAILED_FILE.exists():
            existing_failed = pd.read_csv(
                FAILED_FILE,
                dtype={"provider_id": str},
            )

            if not existing_failed.empty:
                existing_failed["provider_id"] = (
                    normalize_provider_id_series(
                        existing_failed["provider_id"]
                    )
                )

            combined_failed = pd.concat(
                [
                    existing_failed,
                    current_failed,
                ],
                ignore_index=True,
                sort=False,
            )
        else:
            combined_failed = current_failed

        combined_failed = (
            combined_failed
            .dropna(subset=["provider_id"])
            .drop_duplicates(
                subset=["provider_id"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        combined_failed.to_csv(
            FAILED_FILE,
            index=False,
        )


def main() -> None:
    all_hospitals = validate_input_file()

    hospitals, skipped_count = filter_unscraped_hospitals(
        all_hospitals
    )

    print("=" * 80)
    print("MEDICARE HOSPITAL PROFILE SCRAPER")
    print("=" * 80)
    print(
        f"Hospital URLs available: {len(all_hospitals)}"
    )
    print(
        f"Already successfully scraped: {skipped_count}"
    )
    print(
        f"Remaining hospitals to scrape: {len(hospitals)}"
    )

    if hospitals.empty:
        print(
            "\nNo new hospitals need to be scraped. "
            "medicare_raw.csv is already up to date."
        )
        return

    driver = create_driver(
        headless=HEADLESS
    )

    successful_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    try:
        for index, row in hospitals.iterrows():
            provider_id = str(
                row["provider_id"]
            ).strip()

            hospital_name = str(
                row["hospital_name"]
            ).strip()

            profile_url = str(
                row["profile_url"]
            ).strip()

            print("\n" + "=" * 80)
            print(
                f"[{index + 1}/{len(hospitals)}] "
                f"Scraping: {hospital_name}"
            )
            print(f"Provider ID: {provider_id}")
            print(f"URL: {profile_url}")

            try:
                record = scrape_hospital_profile(
                    driver=driver,
                    provider_id=provider_id,
                    expected_name=hospital_name,
                    profile_url=profile_url,
                )

                successful_records.append(record)

                print(
                    f"  Overall rating: "
                    f"{record['overall_rating']}"
                )
                print(
                    f"  Patient survey rating: "
                    f"{record['patient_survey_rating']}"
                )
                print(
                    f"  Hospital type: "
                    f"{record['hospital_type']}"
                )
                print(
                    f"  Emergency services: "
                    f"{record['emergency_services']}"
                )
                print("  Status: success")

            except TimeoutException as error:
                message = (
                    f"TimeoutException: {error}"
                )

                print(
                    f"  Status: failed — {message}"
                )

                failed_records.append({
                    "provider_id": provider_id,
                    "hospital_name": hospital_name,
                    "profile_url": profile_url,
                    "scraping_status": "failed",
                    "scraping_error": message,
                    "scraped_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

            except Exception as error:
                message = (
                    f"{type(error).__name__}: {error}"
                )

                print(
                    f"  Status: failed — {message}"
                )

                failed_records.append({
                    "provider_id": provider_id,
                    "hospital_name": hospital_name,
                    "profile_url": profile_url,
                    "scraping_status": "failed",
                    "scraping_error": message,
                    "scraped_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

            # Preserve the dataset after every profile.
            save_progress(
                successful_records,
                failed_records,
            )

            time.sleep(
                WAIT_BETWEEN_HOSPITALS
            )

    finally:
        driver.quit()

    save_progress(
        successful_records,
        failed_records,
    )

    total_saved = 0

    if OUTPUT_FILE.exists():
        saved_dataframe = pd.read_csv(
            OUTPUT_FILE,
            dtype={"provider_id": str},
        )

        total_saved = len(saved_dataframe)

    print("\n" + "=" * 80)
    print("SCRAPING COMPLETED")
    print("=" * 80)
    print(
        f"Previously completed hospitals skipped: "
        f"{skipped_count}"
    )
    print(
        f"Successfully scraped during this run: "
        f"{len(successful_records)}"
    )
    print(
        f"Failed during this run: "
        f"{len(failed_records)}"
    )
    print(
        f"Total hospitals in medicare_raw.csv: "
        f"{total_saved}"
    )

    if successful_records:
        print(
            f"Dataset saved to: {OUTPUT_FILE}"
        )

    if failed_records:
        print(
            f"Failed records saved to: {FAILED_FILE}"
        )

    print(
        f"Debug files saved under: "
        f"{DEBUG_DIRECTORY}"
    )


if __name__ == "__main__":
    main()