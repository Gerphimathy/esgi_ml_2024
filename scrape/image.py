import os
import requests

q = set()

if os.path.isfile('movies.csv'):
    # load csv data into queue and urls set
    with open('movies.csv', 'r') as f:
        for line in f.readlines():
            if line.startswith('slug'):
                continue
            slug = line.split(';')[0]
            url = line.split(";")[2]
            q.add((slug, url))


i = 0

for slug, url in q:
    i += 1

    if os.path.isfile(f'images/{slug}.jpg'):
        continue

    print(f'Downloading: {i} / {q.__len__()}')

    try:
        resp = requests.get(url)
    except:
        print(f'Error: on {url}')
        continue

    if resp.status_code != 200:
        print(f'Error: {resp.status_code} on {url}')
        continue

    with open(f'images/{slug}.jpg', 'wb') as f:
        f.write(resp.content)