using Microsoft.ML;
using Microsoft.ML.Data;
using IrisFlowerClustering;
using static Microsoft.ML.DataOperationsCatalog;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
string _clusteringModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
string _multiclassModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisMulticlassModel.zip");

MLContext mlContext = new MLContext(seed: 0);
TrainTestData splitDataView = LoadData(mlContext);
var pipeline = ProcessData(mlContext);

ITransformer clusteringModel = ClusteringTrain(mlContext, splitDataView.TrainSet, pipeline);
ClusteringEvaluate(mlContext, clusteringModel, splitDataView.TestSet);
SaveModelAsFile(mlContext, splitDataView.TrainSet.Schema, clusteringModel, _clusteringModelPath);

ITransformer multiclassModel = MulticlassTrain(mlContext, splitDataView.TrainSet, pipeline);
MulticlassEvaluate(mlContext, multiclassModel, splitDataView.TestSet);
SaveModelAsFile(mlContext, splitDataView.TrainSet.Schema, multiclassModel, _multiclassModelPath);

ITransformer loadedClusteringModel = LoadModelFromFile(mlContext, _clusteringModelPath);
UseClusteringModelWithSingleItem(mlContext, loadedClusteringModel);

ITransformer loadedMulticlassModel = LoadModelFromFile(mlContext, _multiclassModelPath);
UseMulticlassModelWithSingleItem(mlContext, loadedMulticlassModel);

TrainTestData LoadData (MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');
    TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.1);
    return splitDataView;
}

IEstimator<ITransformer> ProcessData (MLContext mlContext)
{
    var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "FlowerType", outputColumnName: "Label")
        .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
        .AppendCacheCheckpoint(mlContext);

    return pipeline;
}

ITransformer ClusteringTrain (MLContext mlContext, IDataView splitTrainSet, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: "Label", outputColumnName: "ExpectedFlowerTypeId"));

    Console.WriteLine("=============== Create and Train the Clustering Model ===============");
    var model = trainingPipeline.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

ITransformer MulticlassTrain(MLContext mlContext, IDataView splitTrainSet, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline
        .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: "Label", outputColumnName: "ExpectedFlowerTypeId"))
        .Append(mlContext.Transforms.Conversion.MapKeyToValue(inputColumnName: "PredictedLabel", outputColumnName: "PredictedFlowerType"));

    Console.WriteLine("=============== Create and Train the Multi-class Model ===============");
    var model = trainingPipeline.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

void ClusteringEvaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Clustering Model accuracy with Test data===============");
    IDataView transformedTestData = model.Transform(splitTestSet);
    ClusteringMetrics metrics = mlContext.Clustering.Evaluate(transformedTestData, labelColumnName: "Label");

    // Convert IDataView object to a list.
    var predictions = mlContext.Data.CreateEnumerable<ClusterPrediction>(transformedTestData, reuseRowObject: false).ToList();

    // Print 5 predictions.
    // Note that the label is only used as a comparison with the predicted label. It is not used during training.
    foreach (var p in predictions.Take(2))
        Console.WriteLine($"Expected flower type: {p.ExpectedFlowerTypeId} - {p.FlowerType}, Predicted cluster: {p.PredictedClusterId}");

    foreach (var p in predictions.TakeLast(3))
        Console.WriteLine($"Expected flower type: {p.ExpectedFlowerTypeId} - {p.FlowerType}, Predicted cluster: {p.PredictedClusterId}");

    Console.WriteLine();
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Clustering Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"Average distance: {metrics.AverageDistance:F2}");
    Console.WriteLine($"Davies Bouldin index: {metrics.DaviesBouldinIndex:F2}");
    Console.WriteLine($"Normalized mutual information: {metrics.NormalizedMutualInformation:F2}");
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine();
}

void MulticlassEvaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Multi-class Model accuracy with Test data===============");
    IDataView transformedTestData = model.Transform(splitTestSet);
    MulticlassClassificationMetrics testMetrics = mlContext.MulticlassClassification.Evaluate(transformedTestData);

    // Convert IDataView object to a list.
    var predictions = mlContext.Data.CreateEnumerable<MulticlassPrediction>(transformedTestData, reuseRowObject: false).ToList();

    // Print 5 predictions.
    // Note that the label is used during training.
    foreach (var p in predictions.Take(2))
        Console.WriteLine($"Expected flower type: {p.ExpectedFlowerTypeId} - {p.FlowerType}, Predicted flower type: {p.PredictedFlowerTypeId} - {p.PredictedFlowerType}");

    foreach (var p in predictions.TakeLast(3))
        Console.WriteLine($"Expected flower type: {p.ExpectedFlowerTypeId} - {p.FlowerType}, Predicted flower type: {p.PredictedFlowerTypeId} - {p.PredictedFlowerType}");

    Console.WriteLine();
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine();
}

void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model, string modelPath)
{
    using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
    {
        mlContext.Model.Save(model, trainingDataViewSchema, fileStream);
    }
}

ITransformer LoadModelFromFile(MLContext mlContext, string modelPath)
{
    ITransformer loadedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
    return loadedModel;
}

void UseClusteringModelWithSingleItem (MLContext mlContext, ITransformer model)
{
    PredictionEngine<IrisData, ClusterPrediction> predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);
    var prediction = predictor.Predict(TestIrisData.Setosa);

    Console.WriteLine();
    Console.WriteLine("=============== Single Prediction ===============");
    Console.WriteLine($"Predicted cluster: {prediction.PredictedClusterId}");
    Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void UseMulticlassModelWithSingleItem(MLContext mlContext, ITransformer model)
{
    PredictionEngine<IrisData, MulticlassPrediction> predictor = mlContext.Model.CreatePredictionEngine<IrisData, MulticlassPrediction>(model);
    var prediction = predictor.Predict(TestIrisData.Setosa);

    Console.WriteLine();
    Console.WriteLine("=============== Single Prediction ===============");
    Console.WriteLine($"Predicted flower type: {prediction.PredictedFlowerTypeId} - {prediction.PredictedFlowerType}");
    Console.WriteLine($"Scores: {string.Join(" ", prediction.Scores ?? Array.Empty<float>())}");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}