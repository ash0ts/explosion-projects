from html import entities
import typer
import spacy
from pydantic import BaseModel
from typing import List, Dict


def load_wandb_model(model_path):
    nlp = spacy.load(model_path)
    return nlp


def predict(nlp, row):
    # TODO: Process prediction response for endpoint here
    txt = row['text']
    doc = nlp(txt)
    ents = []
    for ent in doc.ents:
        ents.append(
            {
                "text": ent.text,
                "label": ent.label_
            }
        )
    pred = {
        "text": txt,
        "entities": ents
    }
    return pred


class Request(BaseModel):
    text: str


class Response(BaseModel):
    text: str
    entities: list[dict]


def main(txt: str):
    nlp = load_wandb_model()
    prediction = predict(nlp, txt)
    typer.echo(prediction)


if __name__ == "__main__":
    typer.run(main)
