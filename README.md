# Agent Training Data Preparation

This folder contains scripts to prepare training data for the QA agent using the [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) dataset.

## Setup

Run the setup script to create a virtual environment and install dependencies:

```bash
bash setup_env.sh
```

## Formatting Data

The `format_mind2web.py` script pulls the Mind2Web dataset and formats it into a JSONL format that matches the agent's input (including aria snapshots with `ref` tags and interaction history).

To run the formatting script:

```bash
source venv/bin/activate
python3 format_mind2web.py
```

The output will be saved to `mind2web_training_data.jsonl`.

### Customization

You can modify `format_mind2web.py`:
- `max_samples`: Increase this to process more samples (default is 200).
- `streaming`: Set to `False` in `load_dataset` if you want to download the entire dataset first for faster processing of large batches.

## Verification

You can verify the formatting logic with a small test script:

```bash
source venv/bin/activate
python3 test_format.py
```
