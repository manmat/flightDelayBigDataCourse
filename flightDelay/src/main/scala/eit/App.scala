package eit

/**
 * @author ${user.name}
 */

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
object App {
  def main(args : Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("My first Spark application").setMaster("local[1]")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("BigData")
      .getOrCreate()

    val data = spark.read.format("org.apache.spark.csv").option("header", "true").csv(args(0))


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
    println("Hello" + relevant.columns)
  }
}