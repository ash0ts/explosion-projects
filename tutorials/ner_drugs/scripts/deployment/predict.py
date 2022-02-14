import wandb
import spacy
import os
from dotenv import load_dotenv, find_dotenv
import typer
from importlib.machinery import SourceFileLoader
import subprocess
import sys

load_dotenv(find_dotenv())

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME")
MODEL_ARTIFACT_NAME = os.environ.get("MODEL_ARTIFACT_NAME")
MODEL_ARTIFACT_VERSION = os.environ.get("MODEL_ARTIFACT_VERSION")

#TODO: Not enforce the model to be hard logged with the name "promoted_model"
def load_model():
    #TODO: Grab information about deployment and pass it into here?
    run = wandb.init(project=WANDB_PROJECT_NAME, name="download-model", job_type="deployment")
    model_art = run.use_artifact(
        f"{WANDB_ENTITY}/{WANDB_PROJECT_NAME}/{MODEL_ARTIFACT_NAME}:{MODEL_ARTIFACT_VERSION}")
    model_art_path = model_art.download()  # Enforce path here?
    model_path = os.path.join(model_art_path, 'promoted_model')
    # deployment_artifacts_path = os.path.join(model_art_path, "deployment")
    nlp = spacy.load(model_path)
    run.finish()
    return nlp


def predict(nlp, txt):
    #TODO: Process prediction response for endpoint here
    return nlp(txt)

# def import_code():
#     deployment = SourceFileLoader("predict", __file__).load_module()
#     return deployment

def install():
    command = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--requirement',
        os.path.join(os.path.dirname(__file__), 'requirements.txt'),
    ]

    subprocess.check_call(command)

def main(txt: str):
    nlp = load_model()
    prediction = predict(nlp, txt)
    typer.echo(prediction)


if __name__ == "__main__":
    typer.run(main)