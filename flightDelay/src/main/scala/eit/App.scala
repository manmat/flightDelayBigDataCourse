package eit

/**
 * @author ${user.name}
 */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, CrossValidator}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.Row
import Array._


object App {
  def main(args : Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)

    val conf = new SparkConf().setAppName("MyFirstSparkApplication").setMaster("local[1]")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("MyFirstSparkApplication")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    val data = spark
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("sep",",")
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val new_data = data.select(
      functions.concat(
        data("Year"),
        functions.lit("-"),
        data("Month"),
        functions.lit("-"),
        data("DayofMonth")
      ).as("date"),
      functions.when(
        data("CRSDepTime") < 1200, "Morning"
      ).otherwise(functions.when(data("CRSDepTime") < 1900, "Afternoon").otherwise("Night")).as("DayTime"),
      data("Origin"),
      data("Dest")
    )

    new_data.show()


    //use cache??
    // make loggin work!!

    // add airport business for origin and destination
    val size_airport_orig = data.groupBy("Origin").count()
    val size_airport_dest = data.groupBy("Dest").count()

    val joined = data.join(size_airport_orig, "Origin")
      .withColumnRenamed("count", "AirportBusinessOrig")
      .join(size_airport_dest, "Dest")
      .withColumnRenamed("count", "AirportBusinessDest")


    // select and convert types of Variables that should be used
    // and filter out values where ArrDelay is null
    val converted = joined.select(
      joined("Month"),
      joined("DayofMonth"),
      joined("UniqueCarrier"),
      joined("DayOfWeek"),
      joined("DepDelay").cast(IntegerType),
      joined("CRSDepTime").cast(IntegerType),
      joined("ArrDelay").cast(DoubleType),
      joined("Distance").cast(DoubleType),
      joined("AirportBusinessDest").cast(DoubleType),
      joined("AirportBusinessOrig").cast(DoubleType),
      joined("CRSElapsedTime").cast(DoubleType),
      joined("TaxiOut").cast(DoubleType)
    ).where("ArrDelay is not null")


    // add vector columns for categorical variables
    val to_index = List("DayOfWeek", "Month", "UniqueCarrier", "DayofMonth")

    val indexers = to_index.map{i => new StringIndexer()
      .setInputCol(i)
      .setOutputCol(i + "Index")
    }

    val encoders = to_index.map{j => new OneHotEncoder()
      .setInputCol(j + "Index")
      .setOutputCol(j + "Vec")
    }

    // assemble features as vector
    val assembler = new VectorAssembler()
      .setInputCols(
        Array("MonthVec", "DayofMonthVec","UniqueCarrierVec","DayOfWeekVec",
          "DepDelay", "CRSDepTime","Distance" , "AirportBusinessDest",
          "AirportBusinessOrig", "CRSElapsedTime", "TaxiOut"))
      .setOutputCol("features")

    // build a linear model
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("ArrDelay")
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    // perform grid search
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 1.0))
      .build()

    val stages : Array[PipelineStage] = concat(indexers.toArray, encoders.toArray, Array(assembler, lr))

    val pipeline = new Pipeline().setStages(stages)

    val tvs = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("ArrDelay"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator().setLabelCol("ArrDelay"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)


    // split into train and test datasets
    val Array(trainingData, testData) = converted.randomSplit(Array(0.7, 0.3), seed = 12546)

    val lrModel = tvs.fit(trainingData)
    //val cvModel = cv.fit(trainingData)
    val predictions = lrModel.transform(testData)
      .select("ArrDelay", "prediction")

    predictions.show

    val metrics = new RegressionMetrics(predictions.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println(s"RMSE: ${metrics.rootMeanSquaredError}")
    println(s"MSE: ${metrics.meanSquaredError}")
    println(s"r2: ${metrics.r2}")
    println(s"Explained Variance: ${metrics.explainedVariance}")

    /*


    // alternative approach
    val output = assembler.transform(trainingData)
    val lrModel2 = lr.fit(output)
    println("Model 1 was fit using parameters: " + lrModel2.parent.extractParamMap)

    // Print the coefficients and intercept for linear regression
    println(lrModel2.featuresCol)
    println(s"Coefficients:")
    lrModel2.coefficients.toArray.foreach(x => println(x))
    println(s"Intercept: ${lrModel2.intercept}")

    val trainingSummary = lrModel2.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
  */

  }
}