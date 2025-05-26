using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MovieRecommender;

string _trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "MovieRecommenderModel.zip");

MLContext mlContext = new MLContext();
(IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);
ITransformer model = BuildAndTrainModel(mlContext, trainingDataView, ProcessData(mlContext));
EvaluateModel(mlContext, testDataView, model);
SaveModel(mlContext, trainingDataView.Schema, model);
UseModelForSinglePrediction(mlContext, model);

(IDataView training, IDataView test) LoadData (MLContext mlContext)
{
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(_trainingDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(_testDataPath, hasHeader: true, separatorChar: ',');

    return (trainingDataView, testDataView);
}

IEstimator<ITransformer> ProcessData (MLContext mlContext)
{
    IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));

    return estimator;
}

ITransformer BuildAndTrainModel (MLContext mlContext, IDataView trainingDataView, IEstimator<ITransformer> estimator)
{
    MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options()
    {
        MatrixColumnIndexColumnName = "userIdEncoded",
        MatrixRowIndexColumnName = "movieIdEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 20,
        ApproximationRank = 100
    };

    var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");
    ITransformer model = trainerEstimator.Fit(trainingDataView);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

void EvaluateModel (MLContext mlContext, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Evaluating the model ===============");
    IDataView prediction = model.Transform(testDataView);
    RegressionMetrics metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
    Console.WriteLine();
}

void UseModelForSinglePrediction (MLContext mlContext, ITransformer model)
{
    Console.WriteLine("=============== Making a prediction ===============");
    PredictionEngine<MovieRating, MovieRatingPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    MovieRating testInput = new MovieRating { userId = 6, movieId = 10 };
    MovieRatingPrediction movieRatingPrediction = predictionEngine.Predict(testInput);

    Console.WriteLine($"Predicted rating: {movieRatingPrediction.Score}");
    
    if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine("Movie " + testInput.movieId + " is recommended for user " + testInput.userId);
    }
    else
    {
        Console.WriteLine("Movie " + testInput.movieId + " is not recommended for user " + testInput.userId);
    }

    Console.WriteLine();
}

void SaveModel (MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    mlContext.Model.Save(model, trainingDataViewSchema, _modelPath);
}