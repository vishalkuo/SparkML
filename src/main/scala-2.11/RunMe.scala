import breeze.linalg.{norm, DenseVector}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by vishalkuo on 2016-01-26.
  */
object RunMe {
  val conf = new SparkConf()
    .setMaster("local[2]").setAppName("Movie Lens Recommender").set("spark.executor.memory", "3g")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  def main (args: Array[String]): Unit = {
    val u = new UserTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.user"))
    val m = new MovieTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.item"))
    val o = new RatingTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.data"))

    //Drop the timestamp field from rating
    val extractedFields = o.ratingFields.map(x => x.take(3))
    //Convert extracted fields into RDD of rating objects and train
    val ratingRDD = produceRating(extractedFields)
    val model = ALS.train(ratingRDD, 50, 10, 0.01)
    val userId = 200
    val nReccomendations = 10
    val topTen = model.recommendProducts(userId, nReccomendations)
//    outputTop(ratingRDD, userId, topTen, m)

    val itemId =567
    val itemFac = model.productFeatures.lookup(itemId).head
    val itemVec = DenseVector(itemFac)
    val similarities = model.productFeatures.map({ case (id, arr) =>
      val facVec = DenseVector(arr)
      val similarity = cosSimilarity(itemVec, facVec)
      (id, similarity)
    })

    val topSimilarities = similarities.top(nReccomendations)(Ordering.by[(Int, Double), Double] {
      x => x._2
    })

    println(topSimilarities.take(10).map(x => (m.idString(x._1), x._2)).mkString("\n"))

  }

  //Produce a rating object with userid, movieid, and actual raitng
  def produceRating(x: RDD[Array[String]]) : RDD[Rating] = {
    x.map({
      x =>  Rating(x(0).toInt, x(1).toInt, x(2).toDouble)
    })
  }

  def predict(userId: Int, movieId: Int, model: MatrixFactorizationModel): Double = {
    model.predict(userId, movieId)
  }

  def cosSimilarity(a: DenseVector[Double], b: DenseVector[Double]): Double = {
    a dot b / (norm(a) * norm(b))
  }

  //Set user as key, find all ratings for user and sort descending
  def outputTop(ratingRDD: RDD[Rating], userId: Int, topTen: Array[Rating],  m:MovieTransform) = {
    val movieForUser = ratingRDD.keyBy(_.user).lookup(userId).sortBy(-_.rating)
    println("Recommended: ")
    movieForUser.take(10).map(rating => (m.idString(rating.product), rating.rating)).foreach(println)
    println("Watched: ")
    topTen.map(rating => (m.idString(rating.product), rating.rating)).foreach(println)
  }
}
