"""
A tiny, fully synthetic "world" of English-like sentences used to train the
educational Transformer of the app (the Transformer tab).

Every fact the model can learn is written here, so the model's knowledge is
completely transparent: it knows capitals, royal titles, a few animals and
places — nothing else.
"""
from __future__ import annotations

import random

COUNTRIES = {
    "france": "paris",
    "italy": "rome",
    "spain": "madrid",
    "germany": "berlin",
    "japan": "tokyo",
    "england": "london",
    "egypt": "cairo",
    "greece": "athens",
    "russia": "moscow",
    "china": "beijing",
}

ROYALS = {"king": "man", "queen": "woman", "prince": "boy", "princess": "girl"}

# Animals and a trait each one typically has (used after "it was").
ANIMALS = {
    "animal": "tired",
    "dog": "tired",
    "cat": "hungry",
    "horse": "scared",
    "bird": "small",
    "cow": "slow",
    "mouse": "scared",
    "lion": "hungry",
}

# Places and a property each one typically has (used after "it was").
PLACES = {"street": "busy", "river": "deep", "road": "wide", "bridge": "old"}

PREDATOR = {"dog": "cat", "cat": "mouse", "lion": "cow", "bird": "fish"}

NUMBERS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

PEOPLE = ["man", "woman", "boy", "girl", "king", "queen"]

GROUPS: dict[str, list[str]] = {
    "country": list(COUNTRIES),
    "capital": list(COUNTRIES.values()),
    "royal": list(ROYALS),
    "person": ["man", "woman", "boy", "girl"],
    "animal": [a for a in ANIMALS if a != "animal"] + ["fish"],
    "place": list(PLACES) + ["city", "country", "palace"],
    "adjective": sorted(set(ANIMALS.values()) | set(PLACES.values())),
    "number": NUMBERS,
}


def sentence(rng: random.Random) -> list[str]:
    """One random sentence as a list of tokens."""
    kind = rng.random()
    if kind < 0.24:
        country, capital = rng.choice(list(COUNTRIES.items()))
        template = rng.random()
        if template < 0.6:
            return ["the", "capital", "of", country, "is", capital, "."]
        if template < 0.85:
            return [capital, "is", "the", "capital", "of", country, "."]
        return [capital, "is", "a", "city", "in", country, "."]
    if kind < 0.36:
        royal, person = rng.choice(list(ROYALS.items()))
        if rng.random() < 0.7:
            return ["the", royal, "is", "a", person, "."]
        other = rng.choice([r for r in ROYALS if r != royal])
        return ["the", royal, "and", "the", other, "live", "in", "the", "palace", "."]
    if kind < 0.62:
        # Ambiguous "it": half of the reasons refer to the animal, half to the place.
        animal, trait = rng.choice(list(ANIMALS.items()))
        place, prop = rng.choice(list(PLACES.items()))
        reason = trait if rng.random() < 0.5 else prop
        return ["the", animal, "didn't", "cross", "the", place, "because", "it", "was", reason, "."]
    if kind < 0.72:
        predator, prey = rng.choice(list(PREDATOR.items()))
        return ["the", predator, "chased", "the", prey, "."]
    if kind < 0.84:
        group = rng.choice(["country", "capital", "animal"])
        word = rng.choice(GROUPS[group])
        noun = {"country": "country", "capital": "city", "animal": "animal"}[group]
        article = "an" if noun == "animal" else "a"
        return [word, "is", article, noun, "."] if group != "animal" else ["the", word, "is", "an", "animal", "."]
    if kind < 0.92:
        start = rng.randrange(0, 6)
        length = rng.randrange(3, 6)
        return NUMBERS[start : start + length]
    person = rng.choice(PEOPLE)
    capital = rng.choice(list(COUNTRIES.values()))
    return ["the", person, "lives", "in", capital, "."]


def corpus(count: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [sentence(rng) for _ in range(count)]


TEMPLATES = [
    "the capital of {country} is {capital} .",
    "{capital} is the capital of {country} .",
    "{capital} is a city in {country} .",
    "the {king|queen|prince|princess} is a {man|woman|boy|girl} .",
    "the {royal} and the {royal} live in the palace .",
    "the {animal} didn't cross the {place} because it was {trait of the animal OR property of the place} .",
    "the {predator} chased the {prey} .",
    "{word} is a country / a city / an animal .",
    "one two three … (counting)",
    "the {person} lives in {capital} .",
]
