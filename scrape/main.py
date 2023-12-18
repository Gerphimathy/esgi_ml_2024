import os
import queue
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

max_scrape = 1000
first_url = 'https://www.rottentomatoes.com/m/leave_no_trace'

with open('movies.csv', 'w') as f:
    f.write('slug;genre;poster;\n')
    movies_to_parse = queue.Queue()
    movies_to_parse.put(first_url)
    urls = set()
    while movies_to_parse.qsize() > 0 and urls.__len__() < max_scrape:
        url = movies_to_parse.get()

        if url in urls:
            continue

        urls.add(url)

        # clear console and print progress
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Parsing {url} ({urls.__len__()} / {max_scrape})')

        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # if we encounter a captcha, we stop the script and ask the user to solve it
        if soup.find('div', {'id': 'recaptcha'}) is not None:
            input('Please solve the captcha and press enter to continue...')
            soup = BeautifulSoup(driver.page_source, 'html.parser')

        # get the movie slug from the url
        slug = url.split('/')[-1]

        # get genre, queryselector: span.genre
        genre = soup.find('span', {'class': 'genre'}).text.strip()
        genre = genre.replace('\n', '').replace(' ', '')

        # get poster, queryselector: #main rt-img src tag
        main = soup.find('div', {'class': 'movie-thumbnail-wrap'})
        rt_img = main.find('rt-img')
        poster = rt_img['src']

        # write to csv
        f.write(f'{slug};{genre};{poster};\n')

        # Grab similar movies and add them to the queue
        recommendations = soup.find('section', {'id': 'recommendations'})
        # Get all a tags and get the href attributes
        for a in recommendations.find_all('a'):
            href = a['href']
            if href.startswith('/m/') and href not in urls:
                movies_to_parse.put(f'https://www.rottentomatoes.com{href}')
