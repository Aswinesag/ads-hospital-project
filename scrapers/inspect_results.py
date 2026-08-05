from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


RESULTS_URL = (
    "https://www.medicare.gov/care-compare/results"
    "?searchType=Hospital"
    "&page=1"
    "&city=Texas%20Hill%20Country"
    "&state=TX"
    "&zipcode="
    "&radius=25"
    "&sort=closest"
)


def main() -> None:
    options = Options()
    options.add_argument("--window-size=1500,900")

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(RESULTS_URL)

        input(
            "Wait until the hospital results appear, "
            "then press Enter in this terminal..."
        )

        print("Current URL:", driver.current_url)
        print("Page title:", driver.title)

        articles = driver.find_elements(By.TAG_NAME, "article")
        print(f"Found {len(articles)} article tags")

        for index, article in enumerate(articles, start=1):
            print("\n" + "=" * 80)
            print(f"ARTICLE {index}")
            print(article.text[:1500])

        if not articles:
            print("\nNo article elements found.")

            headings = driver.find_elements(By.TAG_NAME, "h2")
            print(f"Found {len(headings)} h2 elements")

            for index, heading in enumerate(headings, start=1):
                text = heading.text.strip()

                if text:
                    print(f"H2 {index}: {text}")

            links = driver.find_elements(
                By.CSS_SELECTOR,
                "a[href]"
            )

            print(f"\nFound {len(links)} links")

            for link in links:
                text = link.text.strip()
                href = link.get_attribute("href")

                if (
                    text
                    and href
                    and (
                        "hospital" in href.lower()
                        or "provider" in href.lower()
                    )
                ):
                    print({
                        "text": text,
                        "href": href
                    })

        input("\nPress Enter to close Chrome...")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()