import bs4
from urllib.request import urlopen as req
from bs4 import BeautifulSoup as soup

nydoh = 'https://coronavirus.health.ny.gov/county-county-breakdown-positive-cases'

bypass = {'User-Agent': 'Mozilla/5.0'}

nClient = Request(nydoh, headers=bypass)
nyPage = urlopen(nClient)

site_parse = soup(nyPage, 'lxml')
nyPage.close()

tables = site_parse.find("div", {"class": "wysiwyg--field-webny-wysiwyg-body"})

tags = tables.findAll('tr')

csvfile = "COVID-19_cases_nydoh.csv"
headers = "County, Positive Cases \n"

file = open(csvfile, "w")
file.write(headers)

for tag in tags[1:47]:
    pull = tag.findAll('td')
    print("County = %s, Positive Cases = %s" % \
          (pull[0].text, pull[1].text))
    
    file.write(pull[0].text + ", " + pull[1].text + "\n")

file.close()