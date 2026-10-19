"""Verify full-history navigation in the built Streamlit component using Playwright."""
import argparse
import re

from playwright.sync_api import sync_playwright


def verify(url, channel):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1034})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""
            window.addEventListener('message', ({data}) => {
                const rows = data?.args?.response?.body;
                if (data?.type === 'streamlit:render' && Array.isArray(rows) && rows[0]?.date) {
                    window.__testHistory = rows;
                }
            });
        """)

        def expose_chart(route):
            response = route.fetch()
            # Instrument only the test browser's response; shipped assets stay unchanged.
            script, count = re.subn(
                r"([\w$]+)\.timeScale\(\)\.scrollToRealTime\(\)",
                r"(window.__testChart=\1,\g<0>)",
                response.text(),
            )
            assert count == 1, "Expected one chart initialization in the component bundle"
            route.fulfill(response=response, body=script)

        page.route("**/component/streamlit_app.gkhv/assets/*.js", expose_chart)
        page.goto(url)
        page.locator('iframe[title="streamlit_app.gkhv"]').wait_for()
        frame = next(frame for frame in page.frames if "/component/streamlit_app.gkhv/" in frame.url)
        frame.wait_for_function("window.__testHistory?.length > 500 && window.__testChart", timeout=90000)
        frame.wait_for_function("""() => window.__testChart.timeScale().getVisibleRange()?.to
            === window.__testHistory.at(-1).date""", timeout=10000)
        history = frame.evaluate("({count: __testHistory.length, first: __testHistory[0].date, last: __testHistory.at(-1).date})")
        print(f"Loaded {history['count']} bars: {history['first']} to {history['last']}")
        print("Initial chart shows latest bar: PASS")

        frame.evaluate("window.__testChart.timeScale().setVisibleLogicalRange({from: 30, to: 150})")
        box = frame.locator('.chart-wrapper').bounding_box()
        assert box
        page.mouse.move(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.3)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.3, steps=20)
        page.mouse.up()
        frame.wait_for_function("""() => window.__testChart.timeScale().getVisibleRange()?.from
            === window.__testHistory[0].date""", timeout=10000)
        print("Mouse drag reaches earliest available candle: PASS")
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--channel", default="chromium")
    args = parser.parse_args()
    verify(args.url, args.channel)
