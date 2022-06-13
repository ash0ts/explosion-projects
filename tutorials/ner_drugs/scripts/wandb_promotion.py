from dotenv import find_dotenv, load_dotenv
from utils import flatten_dict, load_config
import wandb
import os
import pandas as pd
from pathlib import Path
import typer
load_dotenv(find_dotenv())

WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME")

# We will be enforcing usage of yaml for this


def main(
    config_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False),
    deployment_dir: Path = typer.Argument(
        ..., exists=True, dir_okay=True)
):
    config = load_config(config_path)

    ENV_TYPE = config.get("deployment.environment", "deployable-models")

    run = wandb.init(project=WANDB_PROJECT_NAME, job_type="promote", tags=[
                     "spacy", "ner", "promote", "drugs"], name=f"promote-drug-ner-spacy-model", config=config)

    # Hardcoding the name of the registry for workflow consistency
    candidate_model_evals_table = run.use_artifact(
        "draft-drug-ner-spacy-models:latest").get("candidate_evals")
    eval_metrics_df = pd.DataFrame(
        candidate_model_evals_table.data, columns=candidate_model_evals_table.columns)

    def min_max_scaling(series):
        return (series - series.min()) / (series.max() - series.min())

    metadata_cols = ["model_id", "model_name",
                     "run_id", "run_url", "displacy_entities"]
    eval_metrics_df["original_speed"] = eval_metrics_df["speed"]
    eval_metrics_df["speed"] = min_max_scaling(
        eval_metrics_df["original_speed"])
    eval_metrics_df["promotion_score"] = 0.0

    cols_to_log = []
    for metric, weight in config["weights"].items():
        eval_metrics_df["promotion_score"] += eval_metrics_df[metric] * weight
        cols_to_log.append(metric)
    cols_to_log.append("promotion_score")
    cols_to_log.extend(metadata_cols)

    promotion_df = eval_metrics_df[cols_to_log]
    # TODO: Add as artifact of promotion details of type records(?)
    promotion_table_art = wandb.Artifact(
        "promotion_details", type=ENV_TYPE)
    promotion_table = wandb.Table(dataframe=promotion_df)
    promotion_table_art.add(promotion_table, "promotion_details")

    # LOGGING FOR DEMONSTRATIVE PURPOSES
    run.log_artifact(promotion_table_art)
    run.log({"promotion_details": promotion_table})

    promoted_candidate_model_details = promotion_df.iloc[promotion_df["promotion_score"].idxmax(
    )]
    promoted_candidate_model_id = promoted_candidate_model_details["model_id"]
    promoted_candidate_model_name = promoted_candidate_model_details["model_name"]
    promoted_candidate_model_artifact = run.use_artifact(
        promoted_candidate_model_id)

    # TODO: Directly log the artifact reference under a different artifact name as opposed to downloading and then logging the contents
    # Better for data lineaging in wandb?
    promoted_candidate_model_artifact_path = promoted_candidate_model_artifact.download()
    promoted_candidate_model_path = Path(
        promoted_candidate_model_artifact_path, promoted_candidate_model_name)

    promoted_model_artifact = wandb.Artifact(
        "drug-ner-spacy-model", ENV_TYPE, metadata=promoted_candidate_model_details.to_dict())
    promoted_model_artifact.add_dir(
        promoted_candidate_model_path, name="model")
    # TODO: Take p
    promoted_model_artifact.add_dir(deployment_dir, name="deployment_assets")
    run.log_artifact(promoted_model_artifact)

    run.finish()

    return None


if __name__ == "__main__":
    typer.run(main)
