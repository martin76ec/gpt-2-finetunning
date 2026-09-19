import json


def load_data():
    data = None
    with open("data/instruction-data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


INSTRUCTION_DATA = load_data()
