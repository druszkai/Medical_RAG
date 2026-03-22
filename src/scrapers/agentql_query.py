"""This script serves as a skeleton template for synchronous AgentQL scripts."""

import logging

import agentql
from agentql.ext.playwright.sync_api import Page
from playwright.sync_api import sync_playwright

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Set the URL to the desired website
URLs = ["https://www.webmd.com/heart-disease/guide-chapter-heart-disease-appointment-prep",
        "https://www.webmd.com/heart-disease/cad",
        "https://www.webmd.com/heart-disease/pad",
        "https://www.webmd.com/hypertension-high-blood-pressure/default.htm",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-overview",
        "https://www.webmd.com/heart/default.htm",
        "https://www.webmd.com/heart-disease/atrial-fibrillation/default.htm",
        "https://www.webmd.com/cholesterol-management/default.htm",
        "https://www.webmd.com/heart/metabolic-syndrome/default.htm",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-symptoms-types",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-tests-diagnosis",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-treatment-care",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-living-with",
        "https://www.webmd.com/hypertension-high-blood-pressure/guide-chapter-hypertension-support-resources",
        ]


def main():
    for URL in URLs:
        with sync_playwright() as p, p.chromium.launch(headless=False) as browser:
            # Create a new page in the browser and wrap it to get access to the AgentQL's querying API
            page = agentql.wrap(browser.new_page())

            # Navigate to the desired URL
            page.goto(URL)

            get_response(page)


def get_response(page: Page):
    query = """
{
  publications[] {
    title
    description
    link
  }
}
    """

    response = page.query_data(query)

    # For more details on how to consume the response, refer to the documentation at https://docs.agentql.com/intro/main-concepts
    print(response)


if __name__ == "__main__":
    main()
