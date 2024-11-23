from graphene import ObjectType, Int, String, Float

class AnimeType(ObjectType):
    anime_id = Int()
    name = String()
    genre = String()
    type = String()
    episodes = Int()
    rating = Float()
    members = Int()

class UserAnimeRatingType(ObjectType):
    user_id = Int()
    anime_id = Int()
    rating = Int()


class BookType(ObjectType):
    ISBN = String()
    book_title = String()
    book_author = String()
    year_of_publication = Int()
    publisher = String()
    image_url_s = String()
    image_url_m = String()
    image_url_l = String()

class UserBookRatingType(ObjectType):
    user_id = Int()
    ISBN = String()
    book_rating = Int()

class UserType(ObjectType):
    user_id = Int()
    location = String()
    age = Int()


class MovieType(ObjectType):
    movie_id = Int()
    title = String()
    genres = String()

class UserMovieRatingType(ObjectType):
    user_id = Int()
    movie_id = Int()
    rating = Float()
    timestamp = String()



VALID_TABLES = {
    "animetable": "animetable",
    "ratings": "UserAnimeRatingType",
    "books_table": "BookType",
    "user_book_ratings": "UserBookRatingType",
    "users_table": "UserType",
    "movies_table": "MovieType",
    "user_movie_ratings": "UserMovieRatingType",
}

VALID_SERIALIZERS = {
    "AnimeType": AnimeType,
    "UserAnimeRatingType": UserAnimeRatingType,
    "BookType": BookType,
    "UserBookRatingType": UserBookRatingType,
    "UserType": UserType,
    "MovieType": MovieType,
    "UserMovieRatingType": UserMovieRatingType,
}