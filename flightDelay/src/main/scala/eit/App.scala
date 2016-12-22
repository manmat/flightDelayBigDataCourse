package eit

/**
 * @author ${user.name}
 */

import org.apache.log4j.{Level, Logger}
import org.apache.hadoop.yarn.util.RackResolver
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.mllib.evaluation.RegressionMetrics
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Row

import Array._


object App {
  def main(args : Array[String]) {
    Logger.getLogger(classOf[RackResolver]).getLevel
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("MyFirstSparkApplication").setMaster("local[1]")
    val sc = new SparkContext(conf)

    sc.setLogLevel("OFF")
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
      joined("Origin"),
      joined("Dest"),
      joined("DepDelay").cast(IntegerType),
      joined("CRSDepTime").cast(IntegerType),
      joined("ArrDelay").cast(DoubleType),
      joined("Distance").cast(DoubleType),
      joined("AirportBusinessDest").cast(DoubleType),
      joined("AirportBusinessOrig").cast(DoubleType),
      joined("CRSElapsedTime").cast(DoubleType),
      joined("TaxiOut").cast(DoubleType),
      functions
        .when(data("CRSDepTime") < 1200 && data("CRSDepTime") > 500, "Morning").otherwise(functions
        .when(data("CRSDepTime") < 1900 && data("CRSDepTime") >= 1200, "Afternoon").otherwise("Night")).as("DayTime")
    ).where("ArrDelay is not null and Cancelled = 0")


    println("DATA UNDERSTANDING: \n")

    // drop columns wich have only one category or are empty

    val to_drop = ArrayBuffer()
    for (col <- converted.columns) {
      val distinct = converted.select(col).distinct().count()
      println("Distinct values " + col + ": " + distinct.toString())
      if( distinct <= 1 ){
        println(s"${col} has 1 distinct value or less and thus will be dropped")
        to_drop :+ col
      }
    }

    val total_count = converted.count()
    val to_filter = StringBuilder.newBuilder
    for (col <- converted.columns diff to_drop) {
      val count = converted.where(col + " is null").count()
      println("Missing values " + col + ": " + count.toString())
      if( (count.toFloat != 0) && (count.toFloat / total_count) < 0.05){
        println(col + " has " + (count.toFloat / total_count)*100 + "% missing values")
        if(to_filter.length > 0){
          to_filter.append(" and ")
        }
        to_filter.append(col + " is not null")
      }
      else{
        to_drop :+ col
      }
    }

    println(to_filter.toString())
    to_drop.foreach(x => println(x))

    if( !to_filter.isEmpty ) {
      val filtered = converted.where(to_filter.toString())
    }
    else{
      val filtered = converted
    }


    converted.describe().show()
    converted.groupBy("DayTime").count().orderBy("count").show()

    println("Correlation between ArrDelay and DepDelay: "
      + converted.stat.corr("ArrDelay", "DepDelay").toString
      + "\n")

    // add vector columns for categorical variables
    val to_index = List("DayOfWeek", "Month", "UniqueCarrier", "DayofMonth", "DayTime")

    val indexers = to_index.map{i => new StringIndexer()
      .setInputCol(i)
      .setOutputCol(i + "Index")
    }

    val encoders = to_index.map{j => new OneHotEncoder()
      .setInputCol(j + "Index")
      .setOutputCol(j + "Vec")
    }

    val final_variables = Array("MonthVec", "DayofMonthVec","UniqueCarrierVec","DayOfWeekVec",
      "DepDelay", "DayTimeVec","Distance", "AirportBusinessDest",
      "AirportBusinessOrig", "CRSElapsedTime", "TaxiOut") diff to_drop

    println("Final Variables: ")
    final_variables.foreach(x => println(x))

    // assemble features as vector
    val assembler = new VectorAssembler()
      .setInputCols(final_variables)
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

    // split into train and test datasets
    val Array(trainingData, testData) = filtered.randomSplit(Array(0.7, 0.3), seed = 12546)

    val lrModel = tvs.fit(trainingData)
    //val cvModel = cv.fit(trainingData)
    val predictions = lrModel.transform(testData)
      .select("ArrDelay", "prediction")

    println("PREDICTIONS: ")
    predictions.show(30)

    val metrics = new RegressionMetrics(predictions.rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("MODEL METRICS: \n")
    println(s"RMSE: ${metrics.rootMeanSquaredError}")
    println(s"MSE: ${metrics.meanSquaredError}")
    println(s"r2: ${metrics.r2}")
    println(s"Explained Variance: ${metrics.explainedVariance}")

  }
}