package eit

/**
 * @author ${user.name}
 */

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
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

    // val sq = SQLContext()

    val data = spark
      .read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("sep",";")
      .option("mode", "DROPMALFORMED")
      .load(args(0))


      // spark.read.format("org.apache.spark.csv").option("header", "true").csv(args(0))
    // path should be preferably hdfs path

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

    val target = "ArrDelay"
    // transform variables for linear regression
    // put into form (target, (other variables))
    // create precision and recall as well as ROC curve? other measures?

    println("Hello" + relevant.columns)
  }
}