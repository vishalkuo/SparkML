import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by vishalkuo on 2016-01-26.
  */
object RunMe {
  val conf = new SparkConf()
    .setMaster("local[2]").setAppName("Movie Lens Recommender").set("spark.executor.memory", "4g")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  def main (args: Array[String]): Unit = {
    val u = new UserTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.user"))
    val m = new MovieTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.item"))
    val o = new RatingTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.data"))


    val extractedFields = o.ratingFields.map(x => x.take(3))

  }
}
