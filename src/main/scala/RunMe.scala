import breeze.linalg.{DenseMatrix, norm, DenseVector}
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by vishalkuo on 2016-01-26.
  */
object RunMe {
  val conf = new SparkConf().setMaster("local[2]").setAppName("Movie Lens Recommender").set("spark.executor.memory", "3g")
  val sc = new SparkContext(conf)
  sc.setLogLevel("WARN")
  def main (args: Array[String]): Unit = {
    val u = new UserTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.user"))
    val m = new MovieTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.item"))
    val o = new RatingTransform(sc.textFile("src/main/resources/datasets/ml-100k/u.data"))
    val rank = 50
    //Drop the timestamp field from rating
    val extractedFields = o.ratingFields.map(x => x.take(3))
    //Convert extracted fields into RDD of rating objects and train
    val ratingRDD = produceRating(extractedFields)
    val model = ALS.train(ratingRDD, rank, 10, 0.01)
    val userId = 789
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

//    println(topSimilarities.take(10).map(x => (m.idString(x._1), x._2)).mkString("\n"))

    val MSE = calculateMSE(ratingRDD, model)
//    outputMSE(MSE)

    val movies = movieForUser(ratingRDD, userId).map(_.product)
    val apkForUser = APK(movies, topTen.map(_.product), 10)
//    println(apkForUser)

    //Calculate MAPK
    val allItemFactors = model.productFeatures.map({
      case(id, factor) => factor
    }).collect()
//    val itemMatrix = DenseMatrix(allItemFactors)
    val itemMatrix = DenseMatrix.zeros[Double](m.numMovies.toInt, rank)
    for (arr <- allItemFactors.zipWithIndex){
      for (innerArr <- arr._1.zipWithIndex){
        itemMatrix(arr._2, innerArr._2) = innerArr._1
      }
    }

    val itemMatrixBcast = sc.broadcast(itemMatrix)

    val allItemRecommendations = model.userFeatures.map({ case(id, featureArray) =>
        val userVector = DenseVector(featureArray)
        val scores = itemMatrixBcast.value * userVector
        val sortedScoreZip  = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedScoreZip.map(_._2 + 1).toSeq
        (userId, recommendedIds)
    })

    val userMovies = ratingRDD.map({case Rating(user, product, rating) =>
      (user, product)
    }).groupBy({
      case(user, product) => user
    })

    val MAPK = allItemRecommendations.join(userMovies).map{ case(id, (recommendedIds, actualIds)) =>
        val actualPrediction = actualIds.map(_._2).toSeq
        APK(actualPrediction, recommendedIds, nReccomendations)
    }.reduce(_ + _) / allItemRecommendations.count()
    //println(s"Mean average precision at $nReccomendations: $MAPK")

    //Spark's MSE
//    outputSparkMSE(ratingRDD, model)
    outputSparkMAP(allItemRecommendations, userMovies)
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
  //Mean squared error
  def calculateMSE(ratings: RDD[Rating], model: MatrixFactorizationModel): Double = {
    val userProd = ratings.map({
      case Rating(user, product, rating) => (user, product)
    })
    val predictions = model.predict(userProd).map({
      case Rating(user, product, rating) => ((user, product), rating)
    })
    val ratingPredAggregate = ratings.map({
      case Rating(user, prod, rating) => ((user, prod), rating)
    }).join(predictions)
    val MSE = ratingPredAggregate.map({
      case ((user, product), (act, pred)) => math.pow(act - pred, 2)
    }).reduce(_ + _) / ratingPredAggregate.count()
    MSE
  }

  def outputMSE(MSE: Double) = {
    println("Mean Squared Error: " + MSE)
    println("RMS Error: " + math.sqrt(MSE))
  }

  def movieForUser(ratingRDD: RDD[Rating], userId: Int): Seq[Rating] = {
    ratingRDD.keyBy(_.user).lookup(userId).sortBy(-_.rating)
  }

  //Average Precision at K
  def APK(actual: Seq[Int], predictions: Seq[Int], kVal: Int) : Double = {
    val predictedK = predictions.take(kVal)
    var score = 0.0
    var matches = 0.0
    for ((prediction, index) <- predictedK.zipWithIndex){
      if (actual.contains(prediction)){
        matches += 1.0
        score += matches / (index.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else{
      score / math.min(actual.size, kVal).toDouble
    }
  }

  def outputSparkMSE(ratingRDD: RDD[Rating], model: MatrixFactorizationModel) = {
    val userProd = ratingRDD.map({
      case Rating(user, product, rating) => (user, product)
    })
    val predictions = model.predict(userProd).map({
      case Rating(user, product, rating) => ((user, product), rating)
    })
    val ratingPredAgg = ratingRDD.map({
      case Rating(user, prod, rating) => ((user, prod), rating)
    }).join(predictions).map({ case((user, product), (predicted, actual))
      => (predicted, actual)
    })
    val regMet = new RegressionMetrics(ratingPredAgg)
    println(s"Mean Squared Error = ${regMet.meanSquaredError}")
    println(s"Root Mean Squared Error = ${regMet.rootMeanSquaredError}")
  }
  //Mean average precision
  def outputSparkMAP(allRecs: RDD[(Int, Seq[Int])], userMovies: RDD[(Int, Iterable[(Int, Int)])] ) = {
    val predictedActualAgg = allRecs.join(userMovies).map{ case(id, (predicted, actualZipId)) =>
      (predicted.toArray, actualZipId.map(_._2).toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedActualAgg)
    println(s"MAP = " + rankingMetrics.meanAveragePrecision)
  }
}

