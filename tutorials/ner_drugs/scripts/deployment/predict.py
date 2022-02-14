import typer
import spacy


def load_model(model_path):
    nlp = spacy.load(model_path)
    return nlp


def predict(nlp, txt):
    # TODO: Process prediction response for endpoint here
    doc = nlp(txt)
    ents = []
    for ent in doc.ents:
        ents.append(
            {
                "text": ent.text,
                "label": ent.label_
            }
        )
    return ents


def main(txt: str):
    nlp = load_model()
    prediction = predict(nlp, txt)
    typer.echo(prediction)


if __name__ == "__main__":
    typer.run(main)
