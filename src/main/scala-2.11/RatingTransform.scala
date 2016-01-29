import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-26.
  */
 class RatingTransform(rawFields: RDD[String]) {
  val ratingFields = rawFields.map(lines => lines.split("\t"))
  val ratingCount = ratingFields.count()
  val allRatings = ratingFields.map(fields => fields(2).toDouble)
  val maxRating = allRatings.reduce((a, b) => math.max(a, b))
  val minRating = allRatings.reduce((a, b) => math.min(a, b))
  val averageRating = allRatings.fold(0)((acc, i) => acc + i) / ratingCount

  def ratingsPerUser(numUsers: Long): Double = {
    ratingCount / numUsers.toFloat
  }

  def ratingsPerMovie(numMovies: Long): Double = {
    ratingCount / numMovies.toFloat
  }

  val groupedUserRatings = ratingFields.map(fields => (fields(0).toInt, fields(2).toInt)).groupByKey()
  val userRatingsCount = groupedUserRatings.map(fields => (fields._1, fields._2.toArray))
    .map(fields => (fields._1, fields._2.length)).sortBy(_._1)
}
