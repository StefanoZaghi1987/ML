# ML Playground

[![fastai](https://img.shields.io/badge/fastai-PyTorch-orange)](https://www.fast.ai)
[![ML.NET](https://img.shields.io/badge/ML.NET-C%23%2F.NET-512BD4)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A hands-on machine learning playground that explores the same ideas on two stacks:

- **[fastai](https://www.fast.ai)** (Python / PyTorch) — deep learning notebooks from the fast.ai course, covering the full journey from raw data to an exported, deployable model.
- **[ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)** (C# / .NET) — console apps for classic ML tasks: classification, regression, clustering and recommendation.

Every folder is a small, self-contained project that trains a model end to end — load data, train, evaluate, predict — with no framework magic beyond the library being learned.

## What's inside

| Project | Kind | ML task | Model / algorithm |
|---|---|---|---|
| [`IsThisABird`](IsThisABird) | fastai notebook | Image classification (bird vs. forest) | Transfer learning, ResNet-18 |
| [`SavingFastAIModel`](SavingFastAIModel) | fastai notebook | Cat vs. dog classification + model export | Transfer learning, ResNet-18 |
| [`GradioPets`](GradioPets) | fastai notebook | Cat vs. dog classification (Oxford-IIIT Pet) | Transfer learning, ResNet-18 |
| [`GitHubIssueClassification`](GitHubIssueClassification) | ML.NET app (.NET 8) | Multiclass text classification | `SdcaMaximumEntropy` + `FeaturizeText` |
| [`SentimentAnalysis`](SentimentAnalysis) | ML.NET app (.NET 6) | Binary sentiment analysis | `SdcaLogisticRegression` + `FeaturizeText` |
| [`TaxiFarePrediction`](TaxiFarePrediction) | ML.NET app (.NET 6) | Fare regression (NYC taxi trips) | `FastTree` |
| [`MovieRecommender`](MovieRecommender) | ML.NET app (.NET 6) | Movie recommendation | `MatrixFactorization` |
| [`IrisFlowerClustering`](IrisFlowerClustering) | ML.NET app (.NET 6) | Clustering + species classification | `KMeans` + `SdcaMaximumEntropy` |

## fastai notebooks

### [Is it a bird?](IsThisABird/is-it-a-bird-creating-a-model-from-your-own-data.ipynb)

The classic fast.ai lesson 1: building an image classifier from scratch with your own data. Images of "bird" and "forest" are downloaded via DuckDuckGo search, cleaned with `verify_images`, and split with a `DataBlock`. A pretrained ResNet-18 is fine-tuned and reaches near-perfect accuracy on the validation set, then predicts on a downloaded photo.

### [Saving a basic fastai model](SavingFastAIModel/saving-a-basic-fastai-model.ipynb)

The "from model to app" lesson: trains a cat-vs-dog classifier on the Oxford-IIIT Pet dataset, then covers the deployment step — exporting the `Learner` to `model.pkl` so it can be loaded and served by a separate application (e.g. a Gradio demo).

### [Gradio Pets](GradioPets/dogs_cats.ipynb)

A minimal training recipe that produces the `model.pkl` used by a Gradio demo app: the Oxford-IIIT Pet dataset, an `ImageDataLoaders` labeled by filename convention (`is_cat`), ResNet-18 fine-tuning, and `learn.export()`.

## ML.NET apps

All five apps share the same flow — load from `Data/`, build a pipeline, train, print evaluation metrics, then predict on samples. Each saves its trained model to disk and reloads it for prediction.

### [GitHubIssueClassification](GitHubIssueClassification) (.NET 8)

Classifies GitHub issues into area labels (e.g. `aspnetcore`, `ml.net`) from title and description. Text is featurized with `FeaturizeText` and classified with `SdcaMaximumEntropy`; evaluation reports micro/macro accuracy and log loss. After training, the app saves `Models/model.zip`, reloads it, and turns interactive: type an issue title and description to get a predicted area.

### [SentimentAnalysis](SentimentAnalysis) (.NET 6)

Binary positive/negative sentiment analysis on customer reviews (`yelp_labelled.txt`, included in the repo). Reports accuracy, AUC and F1, then predicts interactively from console input plus a small batch of example comments.

### [TaxiFarePrediction](TaxiFarePrediction) (.NET 6)

Predicts New York City taxi fares from trip features (vendor, rate code, passenger count, trip distance, payment type) using `FastTree` regression with one-hot encoded categoricals. Reports R² and RMSE, and predicts the fare for a sample trip.

### [MovieRecommender](MovieRecommender) (.NET 6)

Movie recommendation with matrix factorization over user-movie ratings (MovieLens-derived). Evaluates with RMSE/R², then predicts the rating user 6 would give movie 10 and recommends it if the score clears 3.5.

### [IrisFlowerClustering](IrisFlowerClustering) (.NET 6)

Two models on the classic Iris dataset (included in the repo): a K-means clustering into 3 clusters (evaluated with average distance, Davies-Bouldin, NMI) and a multiclass species classifier (`SdcaMaximumEntropy`). Both are saved to `Data/` and used for single-flower predictions.

## Getting started

### .NET apps

Requires the [.NET SDK](https://dotnet.microsoft.com/download) — .NET 8 for `GitHubIssueClassification`, .NET 6 for the others. From the repo root:

```bash
# projects whose data is included — run directly
dotnet run --project SentimentAnalysis/SentimentAnalysis
dotnet run --project IrisFlowerClustering/IrisFlowerClustering

# projects needing a dataset download first (see next section)
dotnet run --project GitHubIssueClassification/GitHubIssueClassification
dotnet run --project TaxiFarePrediction/TaxiFarePrediction
dotnet run --project MovieRecommender/MovieRecommender
```

### fastai notebooks

Open each notebook in Jupyter, VS Code or Kaggle and run it top to bottom — the first cells install their own dependencies (`fastai`, `duckduckgo_search`, …) and each notebook downloads its data automatically. `IsThisABird` scrapes its images into a local `bird_or_not/` folder; the other two use the fastai-hosted Oxford-IIIT Pet dataset.

## Datasets

Small classic datasets (`iris.data`, `yelp_labelled.txt`) are tracked in the repo. The larger ones are not — download them before running the corresponding app:

| Dataset | Used by | Source |
|---|---|---|
| `issues_train.tsv` / `issues_test.tsv` | GitHubIssueClassification | [dotnet/samples](https://github.com/dotnet/samples/tree/main/machine-learning/tutorials/GitHubIssueClassification/Data) — GitHub issues of the .NET repos |
| `taxi-fare-train.csv` / `taxi-fare-test.csv` | TaxiFarePrediction | [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples/tree/main/datasets) — NYC taxi trips |
| `recommendation-ratings-train.csv` / `-test.csv` | MovieRecommender | [dotnet/machinelearning-samples](https://github.com/dotnet/machinelearning-samples/tree/main/samples/csharp/getting-started/MatrixFactorization_MovieRecommendation/Data) — MovieLens-derived ratings |

From the repo root (any shell with `curl`, e.g. Git Bash):

```bash
curl -L -o GitHubIssueClassification/GitHubIssueClassification/Data/issues_train.tsv \
  https://raw.githubusercontent.com/dotnet/samples/main/machine-learning/tutorials/GitHubIssueClassification/Data/issues_train.tsv
curl -L -o GitHubIssueClassification/GitHubIssueClassification/Data/issues_test.tsv \
  https://raw.githubusercontent.com/dotnet/samples/main/machine-learning/tutorials/GitHubIssueClassification/Data/issues_test.tsv
curl -L -o TaxiFarePrediction/TaxiFarePrediction/Data/taxi-fare-train.csv \
  https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/datasets/taxi-fare-train.csv
curl -L -o TaxiFarePrediction/TaxiFarePrediction/Data/taxi-fare-test.csv \
  https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/datasets/taxi-fare-test.csv
curl -L -o MovieRecommender/MovieRecommender/Data/recommendation-ratings-train.csv \
  https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/samples/csharp/getting-started/MatrixFactorization_MovieRecommendation/Data/recommendation-ratings-train.csv
curl -L -o MovieRecommender/MovieRecommender/Data/recommendation-ratings-test.csv \
  https://raw.githubusercontent.com/dotnet/machinelearning-samples/main/samples/csharp/getting-started/MatrixFactorization_MovieRecommendation/Data/recommendation-ratings-test.csv
```

Trained model files (`model.zip`, `model.pkl`, the Iris and recommender model zips) are also not tracked — every app and notebook regenerates them when run.

## License

[MIT](LICENSE)
