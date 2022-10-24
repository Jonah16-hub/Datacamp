
from numpy import NaN
from bs4 import BeautifulSoup
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import json
import math as m


#%%function which fetchs pfizer digitalprints for  pfizer

pfizer = []
#function of scrapping
async def main():
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=False, slow_mo=2000)
      page = await browser.new_page()

      await page.goto("https://news.google.com/search?q=pfizer&hl=en-US&gl=US&ceid=US%3Aen")
      #to accept the google's cookies
      await page.click("div.VfPpkd-RLmnJb")
      contents = await page.query_selector_all("h3.ipQwMb.ekueJc.RD0gLb")
      for content in contents:
        title = await content.query_selector("a.DY5T1d.RZIKme")
        articles = {}
        articles['title'] = await title.inner_text()
        pfizer.append(articles)
      # return pfizer
      await page.close()
    
  
if __name__ == '__main__':
  asyncio.run(main())
  #serializing list to json
  json_object = json.dumps(pfizer)

  #writing to a file
  with open("gn.json", "w") as outfile:
      outfile.write(json_object)






# print(pfizer)

# items = await page.query_selector_all('.shelfProductTile-information')
#       products = []
#       for item in items:
#         product = {}
#         name = await item.query_selector('.shelfProductTile-descriptionLink')
#         product['product name'] = await name.inner_text()
#         products.append(product)





