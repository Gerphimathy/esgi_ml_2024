import os

genres = {}

if os.path.isfile('movies.csv'):
    # load csv data into queue and urls set
    with open('movies.csv', 'r') as f:
        for line in f.readlines():
            if line.startswith('slug'):
                continue
            g = line.split(';')[1].split(',')
            for genre in g:
                if genre not in genres:
                    genres[genre] = 0
                genres[genre] += 1

# sort genres by count
genres = {k: v for k, v in sorted(genres.items(), key=lambda item: item[1], reverse=True)}
print(genres)