# %% [markdown]
# <a target="_blank" href="https://colab.research.google.com/github/eeg2025/startkit/blob/main/challenge_1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Challenge 1: Cross-Task Transfer Learning
#
# Thin training entrypoint. The actual implementation lives under `challenge1/`
# so data, model, and training logic can evolve independently.

from utils.custom_proxy import install_network_from_env


if __name__ == "__main__":
    install_network_from_env()

    from challenge1 import run_training

    run_training()
