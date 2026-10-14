# EV-estimation

This repository has the data for our paper "Do LLMs have good ML intuition?". We build a dataset of 288 ML ideas on eight MLE-bench Lite tasks, each paired with the real change in score it produced, and then prompt five frontier LLMs to predict the new score after each idea before it is run.

## Tasks

All eight tasks are past Kaggle competitions, and for each one we start from a baseline that scores near the middle of its public leaderboard. The short name in brackets is the one used in the data.

- Spooky author identification (`spooky`), log loss
- Toxic comment classification (`jigsaw`), mean ROC AUC
- Random acts of pizza (`pizza`), ROC AUC
- Tabular playground series, May 2022 (`tps`), ROC AUC
- Leaf classification (`leaf`), log loss
- Dog breed identification (`dogbreed`), log loss
- New York City taxi fare prediction (`taxi`), RMSE
- Denoising dirty documents (`denoise`), RMSE

## Files

- `data/ideas.jsonl` has the 288 ideas, with the idea text, the code diff and the scores before and after.
- `data/predictions.jsonl` has every prediction from every model and setting in the paper.
- `data/implementer_swap.jsonl` has 66 ideas that both Fable 5.1 and Kimi K3 implemented.
- `data/code/` has the code of every solution.
- `prompts/` has the prompts we used.

## Fields

- `writer` is the model that proposed and implemented the idea.
- `prompted_direction` says whether the writer was asked to improve or worsen the score. The predictors never saw it.
- `change_sigma` is the real change in score divided by σ, the spread of the real changes on that task. Positive always means the score got better.
- `predicted_change_sigma` is the predicted change in the same units.
- `in_main_results` marks the 269 ideas used for the main results.

The main results in the paper use `reasoning_level = "max"` and `context = "idea_and_diff"`.

## Example

```python
import json

ideas = [json.loads(line) for line in open("data/ideas.jsonl")]
preds = [json.loads(line) for line in open("data/predictions.jsonl")]

real = {i["id"]: i["change_sigma"] for i in ideas if i["in_main_results"]}
opus = {p["id"]: p["predicted_change_sigma"] for p in preds
        if p["model"] == "Opus 5.5" and p["reasoning_level"] == "max" and p["context"] == "idea_and_diff"}
```

## Notes

The data paths in the code are replaced with `<DATA_ROOT>`, `<TORCH_HOME>` and `<HF_HOME>`. The competition data is not included, and you can get it from [MLE-bench](https://github.com/openai/mle-bench).

## License

[MIT](LICENSE)

## Citation

```bibtex
@article{autolab2026intuition,
  title  = {Do {LLMs} have good {ML} intuition?},
  author = {Autolab AI},
  year   = {2026}
}
```
