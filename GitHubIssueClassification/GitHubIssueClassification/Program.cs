using Microsoft.ML;
using Microsoft.ML.Data;
using GitHubIssueClassification;

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]) ?? ".";
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");
MLContext _mlContext = new MLContext(seed: 0);

IDataView _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);
ITransformer _trainedModel = BuildAndTrainModel(_trainingDataView, ProcessData());

GitHubIssue testIssue = new GitHubIssue() { Title = "WebSockets communication is slow in my machine", Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.." };
UseModelWithSingleItem(_trainedModel, testIssue);

Evaluate(_trainedModel);
SaveModelAsFile(_trainingDataView.Schema, _trainedModel);

ITransformer _loadedModel = LoadModelFromFile();
UseModelWithSingleItem(_loadedModel, new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" });
UseModelWithUserInputItem(_loadedModel);

IEstimator<ITransformer> ProcessData()
{
    var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
        .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
        .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
        .AppendCacheCheckpoint(_mlContext);

    return pipeline;
}

ITransformer BuildAndTrainModel (IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = trainingPipeline.Fit(trainingDataView);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

void Evaluate (ITransformer model)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
    IDataView predictions = model.Transform(testDataView);
    MulticlassClassificationMetrics testMetrics = _mlContext.MulticlassClassification.Evaluate(predictions);

    Console.WriteLine();
    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");
}

void SaveModelAsFile (DataViewSchema trainingDataViewSchema, ITransformer model)
{
    _mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
}

ITransformer LoadModelFromFile()
{
    ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
    return loadedModel;
}

void UseModelWithSingleItem(ITransformer model, GitHubIssue issue)
{
    PredictionEngine<GitHubIssue, IssuePrediction> predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);
    var prediction = predEngine.Predict(issue);

    Console.WriteLine();
    Console.WriteLine("=============== Single Prediction ===============");
    Console.WriteLine($"Title: {issue.Title} | Description: {issue.Description} | Result: {prediction.Area} ");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}

void UseModelWithUserInputItem(ITransformer model)
{
    PredictionEngine<GitHubIssue, IssuePrediction> predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(model);

    Console.Write("Please insert the issue title: ");
    string issueTitle = Console.ReadLine() ?? string.Empty;
    Console.Write("Please insert the issue description: ");
    string issueDescription = Console.ReadLine() ?? string.Empty;
    GitHubIssue issue = new GitHubIssue() { Title = issueTitle, Description = issueDescription };

    var prediction = predEngine.Predict(issue);

    Console.WriteLine();
    Console.WriteLine("=============== Single Prediction ===============");
    Console.WriteLine($"Title: {issue.Title} | Description: {issue.Description} | Result: {prediction.Area} ");
    Console.WriteLine("=============== End of Predictions ===============");
    Console.WriteLine();
}