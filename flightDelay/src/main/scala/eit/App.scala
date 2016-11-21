package eit

/**
 * @author ${user.name}
 */
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
object App {
  def main(args : Array[String]) {
    Logger.getRootLogger().setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("My first Spark application").setMaster("local[1]")
    val sc = new SparkContext(conf)
    val data = sc.textFile(args(0))
    val numAs = data.filter(line => line.contains("a")).count()
    val numBs = data.filter(line => line.contains("A")).count()
    println(s"Lines with a: ${numAs}, Lines with b: ${numBs}")
  }
}