import requests
import re
import csv
from bs4 import BeautifulSoup
data= []
sevens = 0
for i in range(2,90):
    URL = "https://www.ielts-practice.org/category/sample-essays/page/" + str(i)
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    Essay_URLs = soup.find_all("h2")
    links = []
    for tag in Essay_URLs:
        if tag.find('a'):
            links.append(tag.find('a').get('href',''))
    for link in links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")
        contents = soup.find_all("div", class_="entry-inner")
        essay = []
        dataline = []
        for content in contents:
            element = content.find_all("p")
            for e in element:
                essay.append(e.text)
        try:
            score = essay[1].split(' ')[4]
        except:
            continue
        #print(score.replace('.','',1).isdigit())
        dataline = [essay[0],'\n'.join(essay[2:len(essay)-2]),score]
        if(score.replace('.','',1).isdigit()):
            if(score != '7.5' or sevens < 300):
                if(score == '7.5'):
                    sevens+=1
                data.append(dataline)