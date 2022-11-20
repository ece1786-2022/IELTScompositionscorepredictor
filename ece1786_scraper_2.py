import sys

import requests
from bs4 import BeautifulSoup
import json
import csv


scraped_questions = {}

# for band in [4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 8.5, 9]:
for band in [3.5, 3, 2.5, 2]:
    scraped_questions[band] = set([])
    idx = 0
    print(band)
    while len(scraped_questions[band]) <= 100:
        URL = "https://writing9.com/band/{}/{}".format(band, idx)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find(id="__NEXT_DATA__")
        parsed = json.loads(results.prettify().split("\n")[1])
        data = parsed['props']['pageProps']['data']
        for q in data:
            if q['band'] == band:
                scraped_questions[band].add(q['_id'] + "-" + q['slug'])
        idx += 1

with open('IEL.csv', 'a', encoding='UTF8') as f:
    writer = csv.writer(f)
    total_data = []

    for band in scraped_questions:
        print(band)
        for x in scraped_questions[band]:
            url = "https://writing9.com/text/{}".format(x)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
            results = soup.find(id="__NEXT_DATA__")
            parsed = json.loads(results.prettify().split("\n")[1])
            try:
                question = parsed['props']['pageProps']['text']['question']
                answer = parsed['props']['pageProps']['text']['text']
                # total_data.append([question, answer, band])
                writer.writerow([question, answer, "<4"])
            except Exception as e:
                print(e)
                sys.exit()

