from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def create_driver() -> webdriver.Chrome:
    options = Options()

    # Keep the browser visible during initial testing.
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")

    # Selenium Manager handles the ChromeDriver setup.
    return webdriver.Chrome(options=options)


def main() -> None:
    driver = None

    try:
        driver = create_driver()

        driver.get("https://www.medicare.gov/care-compare/")

        print("Page title:", driver.title)
        print("Current URL:", driver.current_url)

        body = driver.find_element(By.TAG_NAME, "body")
        body_text = body.text.strip()

        print("\nFirst 500 characters of page text:")
        print(body_text[:500])

        output_directory = Path("data")
        output_directory.mkdir(parents=True, exist_ok=True)

        screenshot_path = output_directory / "selenium_test.png"
        html_path = output_directory / "selenium_test.html"

        driver.save_screenshot(str(screenshot_path))

        html_path.write_text(
            driver.page_source,
            encoding="utf-8"
        )

        print(f"\nScreenshot saved to: {screenshot_path}")
        print(f"HTML saved to: {html_path}")
        print("\nSelenium browser test completed successfully.")

        input("\nPress Enter to close Chrome...")

    except Exception as error:
        print("\nSelenium test failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

        raise

    finally:
        if driver is not None:
            driver.quit()


if __name__ == "__main__":
    main()