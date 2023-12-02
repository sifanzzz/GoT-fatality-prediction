from src.short_results import short_results
import pickle
import click


with open("./test/model.pkl", "rb") as f:
              random_search = pickle.load(f)

def main(ranom_search)
    results = short_results(random_search)
    return results