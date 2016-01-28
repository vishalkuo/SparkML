import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-26.
  */
class UserTransform(sc: SparkContext, rawFields: RDD[String]) {
  val userFields = rawFields.map(line => line.split("\\|"))
  val totalUsers = userFields.count()
  val genders = userFields.map(fields => fields(2)).distinct()
  val declaredOccupations = userFields.map(fields => fields(3)).distinct().count()
  val numOfZipcodes = userFields.map(fields => fields(4)).distinct().count()
  val allAges = userFields.map(fields => fields(1).toInt)
  val occupationsIndexed = userFields.map(fields => fields(3)).distinct().collect()
  val occupationMap = occupationsIndexed.zipWithIndex.toMap


  def getAgeDistribution: RDD[(Int, Int)] = {
    userFields.map(fields => (fields(1).toInt, 1)).reduceByKey((a, b) => a + b).sortBy(_._1)
  }

  def getOccupationDistribution: RDD[(String, Int)] = {
    userFields.map(fields => (fields(3), 1)).reduceByKey(_ + _).sortBy(_._2)
  }
  
}
