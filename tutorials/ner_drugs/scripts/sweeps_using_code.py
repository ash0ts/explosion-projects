import os
import typer
from pathlib import Path
from spacy.training.loop import train
from spacy.training.initialize import init_nlp
from spacy import util
from thinc.api import Config
import wandb
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME")


def main(default_config: Path, output_path: Path):
    def train_spacy():
        loaded_local_config = util.load_config(default_config)
        run = wandb.init(project=WANDB_PROJECT_NAME, job_type="train", tags=[
                         "spacy", "ner", "train", "drugs"], name=f"train-drug-ner-spacy-model")
        sweeps_config = Config(util.dot_to_dict(run.config))
        merged_config = Config(loaded_local_config).merge(sweeps_config)
        nlp = init_nlp(merged_config)
        output_path.mkdir(parents=True, exist_ok=True)
        train(nlp, output_path, use_gpu=True)
        run.finish()

    sweep_config = {"method": "bayes"}
    metric = {"name": "ents_f", "goal": "maximize"}
    sweep_config["metric"] = metric
    parameters_dict = {
        "training.dropout": {
            "min": 0.05,
            "max": 0.5,
        },
        "training.optimizer.learn_rate": {"min": 0.001, "max": 0.01},
        "components.ner.model.maxout_pieces": {"values": [1, 2, 3]},
        "components.ner.model.hidden_width": {"values": [32, 64, 128]},
    }
    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=WANDB_PROJECT_NAME)
    wandb.agent(sweep_id, train_spacy, count=20)


if __name__ == "__main__":
    typer.run(main)
