using Microsoft.ML;
using Microsoft.ML.Data;
using TaxiFarePrediction;

string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
MLContext _mlContext = new MLContext(seed: 0);

var model = Train(_mlContext);
Evaluate(_mlContext, model);
TestSinglePrediction(_mlContext, model);

ITransformer Train (MLContext mlContext)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_trainDataPath, hasHeader: true, separatorChar: ',');

    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
        .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded"))
        .Append(mlContext.Regression.Trainers.FastTree());

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = pipeline.Fit(dataView);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();
    return model;
}

void Evaluate (MLContext mlContext, ITransformer model)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
    IDataView predictions = model.Transform(dataView);
    RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

void TestSinglePrediction (MLContext mlContext, ITransformer model)
{
    PredictionEngine<TaxiTrip, TaxiTripFarePrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

    var taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0 // To predict. Actual/Observed = 15.5
    };

    var prediction = predictionFunction.Predict(taxiTripSample);

    Console.WriteLine();
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine();
}