# TODO: Remove dependency on pandas

import pandas as pd
import os
from pathlib import Path

import typer
import wandb
from dotenv import find_dotenv, load_dotenv
from spacy.cli.evaluate import evaluate
from tqdm import tqdm

from utils import flatten_dict

load_dotenv(find_dotenv())


WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME")


def main(eval_dataset_name: str = "drugs_eval"):

    api = wandb.Api()
    candidate_runs = api.runs(WANDB_ENTITY + "/" + WANDB_PROJECT_NAME)
    config = {}
    run = wandb.init(project=WANDB_PROJECT_NAME, job_type="eval", tags=[
                     "spacy", "ner", "eval", "drugs"], name=f"evaluate-drug-ner-spacy-model", config=config)
    eval_dataset_art = run.use_artifact("dataset:latest").get_path(
        f"dataset/{eval_dataset_name}.spacy")
    eval_data_path = eval_dataset_art.download()

    # TODO: Add args for filtering like based on time date
    table_data = []
    for candidate_run in tqdm(candidate_runs):
        has_model = False
        alias = "latest"

        # See if there exists a checkpoint model type which is the default logged artifact type by Spacy
        for logged_art in candidate_run.logged_artifacts():
            if "checkpoint" == logged_art.type:
                has_model = True
            if "best" in logged_art.aliases:
                alias = "best"

        # Only apply evaluation to logged spacy models with this data
        if has_model:
            candidate_run_id = candidate_run.id
            candidate_run_url = candidate_run.url
            candidate_pipeline_name = f"pipeline_{candidate_run_id}"
            candidate_run_best_model_name = f"{candidate_pipeline_name}:{alias}"
            candidate_run_model_art = run.use_artifact(
                candidate_run_best_model_name)
            candidate_model_path = Path(
                candidate_run_model_art.download(), candidate_pipeline_name)

            displacy_path = Path(".", "displacy")
            if not displacy_path.exists():
                # Force overwrite?
                displacy_path.mkdir(parents=True, exist_ok=False)
            metrics = evaluate(candidate_model_path,
                               eval_data_path, displacy_path=displacy_path)

            metrics["model_id"] = candidate_run_best_model_name
            metrics["model_name"] = candidate_pipeline_name
            metrics["run_id"] = candidate_run_id
            metrics["run_url"] = candidate_run_url

            parse_path = Path(displacy_path, "parses.html")
            if parse_path.exists():
                parse_html = wandb.Html(str(parse_path))
                metrics["displacy_parses"] = parse_html

            ent_path = Path(displacy_path, "entities.html")
            if ent_path.exists():
                ent_html = wandb.Html(str(ent_path))
                metrics["displacy_entities"] = ent_html

            table_data.append(flatten_dict(metrics))

    candidate_model_registry = wandb.Artifact(
        "draft-drug-ner-spacy-models", type="draft-models")
    candidate_model_evaluation_table = wandb.Table(
        dataframe=pd.DataFrame(table_data))
    candidate_model_registry.add(
        candidate_model_evaluation_table, "candidate_evals")

    run.log_artifact(candidate_model_registry)
    # THIS IS FOR DEMO PURPOSES
    run.log({"candidate_model_registry": candidate_model_evaluation_table})
    run.finish()

    return None


if __name__ == "__main__":
    typer.run(main)
