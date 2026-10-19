"""Run against a local Streamlit server with Playwright installed."""
import argparse

from playwright.sync_api import sync_playwright


def verify(url, channel):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel, headless=True)
        page = browser.new_page()
        page.goto(url)
        iframe = page.locator('iframe[title="streamlit_app.gkhv"]')
        iframe.wait_for()
        for width, height in ((1440, 1034), (390, 844), (1440, 650)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_function("""() => {
                const frame = document.querySelector('iframe[title="streamlit_app.gkhv"]');
                const box = frame?.getBoundingClientRect();
                return box && Math.abs(box.top) <= 1
                    && Math.abs(box.bottom - window.innerHeight) <= 1;
            }""", timeout=10000)
            content = iframe.content_frame
            content.locator('.app-grid').wait_for()
            grid = content.locator('.app-grid').bounding_box()
            assert grid and abs(grid["height"] - height) <= 1, grid
            print(f"Viewport {width}x{height}: PASS")
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--channel", default="chromium")
    args = parser.parse_args()
    verify(args.url, args.channel)
