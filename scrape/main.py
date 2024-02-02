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

max_scrape = 40000
first_url = 'https://www.rottentomatoes.com/m/leave_no_trace'
no_poster = "https://images.fandango.com/cms/assets/5d84d010-59b1-11ea-b175-791e911be53d--rt-poster-defaultgif.gif"

movies_to_parse = queue.Queue()
urls = set()

if os.path.isfile('movies.csv'):
    # load csv data into queue and urls set
    with open('movies.csv', 'r') as f:
        for line in f.readlines():
            if line.startswith('slug'):
                continue
            slug = line.split(';')[0]
            url = f'https://www.rottentomatoes.com/m/{slug}'
            urls.add(url)
            first_url = f'https://www.rottentomatoes.com/m/{slug}'

movies_to_parse.put(first_url)

with open('movies.csv', 'a') as f:
    #f.write('slug;genre;poster;\n')
    while movies_to_parse.qsize() > 0 and urls.__len__() < max_scrape:
        url = movies_to_parse.get()

        if url in urls and url != first_url:
            continue

        urls.add(url)

        # clear console and print progress
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'Parsing {url} ({urls.__len__()} / {max_scrape} | {movies_to_parse.qsize()} in queue)')

        try:
            driver.get(url)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
        except:
            continue

        # if we encounter a captcha, we stop the script and ask the user to solve it
        if soup.find('div', {'id': 'recaptcha'}) is not None:
            input('Please solve the captcha and press enter to continue...')
            soup = BeautifulSoup(driver.page_source, 'html.parser')

        # get the movie slug from the url
        slug = url.split('/')[-1]

        # get genre, queryselector: span.genre
        try:
            genre = soup.find('span', {'class': 'genre'}).text.strip()
            genre = genre.replace('\n', '').replace(' ', '')
        except:
            continue

        # get poster, queryselector: #main rt-img src tag
        try:
            main = soup.find('div', {'class': 'movie-thumbnail-wrap'})
            rt_img = main.find('rt-img')
            if rt_img == no_poster:
                continue
            poster = rt_img['src']
        except:
            continue

        # write to csv
        f.write(f'{slug};{genre};{poster};\n')

        try:
            # Grab similar movies and add them to the queue
            recommendations = soup.find('section', {'id': 'recommendations'})
        except:
            continue

        # Get all a tags and get the href attributes
        try:
            for a in recommendations.find_all('a'):
                href = a['href']
                if href.startswith('/m/') and href not in urls:
                    movies_to_parse.put(f'https://www.rottentomatoes.com{href}')
        except:
            continue
