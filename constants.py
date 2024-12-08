from enum import Enum


class InputLength(Enum):
    Bert4Rec = (10, 50)


# The minimum and maximum length that a sequence may be during the genetic generation
MIN_LENGTH, MAX_LENGTH = InputLength.Bert4Rec.value
PADDING_CHAR = -1


cat2id = {
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
}

