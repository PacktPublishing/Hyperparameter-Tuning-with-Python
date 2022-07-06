import optuna

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

run = neptune.init(
    project="louisowen6/hpo-test-1",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzZjMwNTFmNi0xMDhkLTQ4NjYtOTNjZi04YTgyMTZlNTlkZTAifQ==",
)  # your credentials

params = {"direction": "minimize", "n_trials": 20}
run["parameters"] = params


def objective(trial):
    param = {
        "epochs": trial.suggest_int("epochs", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_uniform("dropout", 0.2, 0.8),
    }

    loss = (param["dropout"] * param["learning_rate"]) ** param["epochs"]

    return loss


neptune_callback = optuna_utils.NeptuneCallback(run)

study = optuna.create_study(direction=params["direction"])
study.optimize(objective, n_trials=params["n_trials"], callbacks=[neptune_callback])

run.stop()