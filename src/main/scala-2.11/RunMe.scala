import org.apache.spark.{SparkContext, SparkConf}
import RatingTransform._
/**
  * Created by vishalkuo on 2016-01-26.
  */
object RunMe {
  val conf = new SparkConf()
    .setMaster("local[2]").setAppName("Movie Lens Recommender").set("spark.executor.memory", "1g")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  def main (args: Array[String]): Unit = {
    val u = new UserTransform(sc, sc.textFile("src/main/resources/datasets/ml-100k/u.user"))
    val m = new MovieTransform(sc, sc.textFile("src/main/resources/datasets/ml-100k/u.item"))

//    for (item <- m.getAgeAggregated.take(5)){
//      print(s"${item._1} ${item._2}")
//    }
//
//    for (item <- m.getYearsFilled.take(5)){
//      println(item)
//    }

    m.years.take(5).foreach(println)
  }
}
