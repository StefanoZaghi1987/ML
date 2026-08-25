using Microsoft.ML.Data;

namespace IrisFlowerClustering
{
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength { get; set; }

        [LoadColumn(1)]
        public float SepalWidth { get; set; }

        [LoadColumn(2)]
        public float PetalLength { get; set; }

        [LoadColumn(3)]
        public float PetalWidth { get; set; }

        [LoadColumn(4)]
        public string? FlowerType { get; set; }
    }

    public class ClusterPrediction : IrisData
    {
        [ColumnName("Label")]
        public uint ExpectedFlowerTypeId { get; set; }

        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId { get; set; }

        [ColumnName("Score")]
        public float[]? Distances { get; set; }
    }

    public class MulticlassPrediction : IrisData
    {
        [ColumnName("Label")]
        public uint ExpectedFlowerTypeId { get; set; }

        [ColumnName("PredictedLabel")]
        public uint PredictedFlowerTypeId { get; set; }

        [ColumnName("PredictedFlowerType")]
        public string? PredictedFlowerType { get; set; }

        [ColumnName("Score")]
        public float[]? Scores { get; set; }
    }
}