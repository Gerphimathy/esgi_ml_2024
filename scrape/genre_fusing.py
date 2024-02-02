import os

drama_genres = [
    "Drama",
    "Romance",
    "Documentary",
    "Biography",
    "Lgbtq+", "Gay&lesbian",
    "Biography",
    "History",
]

comedy_genres = [
    "Comedy",
    "Kids&family",
    "Musical",
    "Animation",
    "Holiday",
    "Music",
    "Anime",
    "Sports", "Sports&fitness",
    "Short",
]

action_genres = [
    "Mystery&thriller",
    "Action",
    "Horror",
    "Adventure",
    "Crime",
    "Fantasy",
    "Sci-fi",
    "Western",
    "War",
]
movies = {}

if os.path.isfile('movies.csv'):
    # load csv data into queue and urls set
    with open('movies.csv', 'r') as f:
        for line in f.readlines():
            if line.startswith('slug'):
                continue
            g = line.split(';')[1].split(',')
            s = line.split(';')[0]

            movies[s] = set()

            for genre in g:
                if genre in drama_genres:
                    movies[s].add("Drama")
                if genre in comedy_genres:
                    movies[s].add("Comedy")
                if genre in action_genres:
                    movies[s].add("Action")


with open('dataset.csv', 'w') as f:
    f.write('slug;genres;\n')
    for slug, genres in movies.items():
        f.write(f'{slug};{",".join(genres)};\n')