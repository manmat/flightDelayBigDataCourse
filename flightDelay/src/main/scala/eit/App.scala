package eit

/**
 * @author ${user.name}
 */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j.{Level, Logger}
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

    val relevant = data.drop(
      "ArrTime",
      "ActualElapsedTime",
      "AirTime",
      "TaxiIn",
      "Diverted",
      "CarrierDelay",
      "WeatherDelay",
      "NASDelay",
      "SecurityDelay",
      "LateAircraftDelay"
    )


    // add vector columns for categorical variables

    val to_index = List("DayOfWeek", "Month", "UniqueCarrier", "DayofMonth")

    val indexers = to_index.map{i => new StringIndexer()
      .setInputCol(i)
      .setOutputCol(i + "Index")
      .fit(relevant)
    }
    val encoders = to_index.map{i => new OneHotEncoder()
      .setInputCol(i + "Index")
      .setOutputCol(i + "Vec")
    }

    val collection :Array[PipelineStage] = concat(indexers.toArray, encoders.toArray)

    val pipeline_encode = new Pipeline()
      .setStages(collection)

    val encoded = pipeline_encode.fit(relevant).transform(relevant)


    // add airport business for origin and destination
    val size_airport_orig = encoded.groupBy("Origin").count()
    val size_airport_dest = encoded.groupBy("Dest").count()

    val joined = encoded.join(size_airport_orig, "Origin")
      .withColumnRenamed("count", "AirportBusinessOrig")
      .join(size_airport_dest, "Dest")
      .withColumnRenamed("count", "AirportBusinessDest")

    // drop forbidden columns
    val converted = joined.select(
      joined("MonthVec"),
      joined("DayofMonthVec"),
      joined("UniqueCarrierVec"),
      joined("DayOfWeekVec"),
      joined("DepDelay").cast(IntegerType),
      joined("DepTime").cast(IntegerType),
      joined("ArrDelay").cast(DoubleType),
      joined("Distance").cast(DoubleType),
      joined("AirportBusinessDest").cast(DoubleType),
      joined("AirportBusinessOrig").cast(DoubleType),
      joined("CRSElapsedTime").cast(DoubleType)
    ).where("ArrDelay is not null")


    val Array(trainingData, testData) = converted.randomSplit(Array(0.7, 0.3))

    val assembler = new VectorAssembler()
     .setInputCols(
       Array("MonthVec", "DayofMonthVec","UniqueCarrierVec","DayOfWeekVec",
         "DepDelay", "DepTime", "ArrDelay","Distance", "AirportBusinessDest",
         "AirportBusinessOrig", "CRSElapsedTime"))
     .setOutputCol("features")


    val lr = new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("ArrDelay")
        .setMaxIter(10)
        .setElasticNetParam(0.8)

    val pipeline_model = new Pipeline()
      .setStages(Array(assembler, lr))

    val lrModel = pipeline_model.fit(trainingData)
    lrModel
    println(lrModel)
    lrModel.transform(testData).show

    //val lrModel = lr.fit(output)
/*
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")

    lrModel.coefficients.toArray.foreach(x => println(x))
      */
    // validation measures
}
}