﻿using Microsoft.ML;
using Microsoft.ML.Data;
using SentimentAnalysis;
using static Microsoft.ML.DataOperationsCatalog;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "yelp_labelled.txt");
MLContext mlContext = new MLContext();
TrainTestData splitDataView = LoadData(mlContext);
ITransformer model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);
Evaluate(mlContext, model, splitDataView.TestSet);
UseModelWithSingleItem(mlContext, model);
UseModelWithBatchItems(mlContext, model);

TrainTestData LoadData (MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

ITransformer BuildAndTrainModel (MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

void Evaluate (MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}

void UseModelWithSingleItem (MLContext mlContext, ITransformer model)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    Console.Write("Please insert a comment: ");
    string sentimentText = Console.ReadLine() ?? string.Empty;
    SentimentData sampleStatement = new SentimentData() { SentimentText = sentimentText };

    var resultPrediction = predictionFunction.Predict(sampleStatement);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

    Console.WriteLine();
    Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");

    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void UseModelWithBatchItems (MLContext mlContext, ITransformer model)
{
    IEnumerable<SentimentData> sentiments = new List<SentimentData>()
    {
        new SentimentData() { SentimentText = "This was a horrible meal" },
        new SentimentData() { SentimentText = "I love this spaghetti." }
    };

    IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
    IDataView predictions = model.Transform(batchComments);
    IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

    Console.WriteLine();
    Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");

    foreach (SentimentPrediction prediction in predictedResults)
    {
        Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");
    }

    Console.WriteLine("=============== End of predictions ===============");
}