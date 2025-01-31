import pandas as pd
from playwright.sync_api import sync_playwright, Playwright
import re

# Initialize playwright
pw = sync_playwright().start()
chrome = pw.chromium.launch(headless = False)
page = chrome.new_page()
page.goto('https://www.nflmockdraftdatabase.com/')

# Always run
page.close()
chrome.close()
pw.stop()