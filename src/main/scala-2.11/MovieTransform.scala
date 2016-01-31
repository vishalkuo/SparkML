import breeze.linalg.{normalize, SparseVector}
import breeze.linalg.normalize
import org.apache.spark.broadcast.Broadcast
//import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-27.
  */
class MovieTransform(rawFields: RDD[String]) extends Serializable{
  val movieFields = rawFields.map(lines => lines.split("\\|"))
  val numMovies = movieFields.count()
  val idString = movieFields.map(fields => (fields(0).toInt, fields(1))).collectAsMap()
  val years = movieFields.map(fields => fields(2)).map({
    field => try{
      field.substring(field.length - 4).toInt
    } catch {
      case _: NumberFormatException | _: StringIndexOutOfBoundsException => 1900
    }
  })

  val yearsFiltered = years.filter(_ != 1900)
  val yearsMean = yearsFiltered.mean()

  val yearsFilled = years.map(x => if (x == 1900) yearsMean else x)

  def getAgeAggregated: Array[(Double, Long)] = {
    yearsFilled.map(x => 1998 - x).countByValue().toArray.sortBy(_._1)
  }

  val rawTitles = movieFields.map(fields => fields(1))
  val titleFiltered = rawTitles.map(title => title.replaceAll("\\((\\w+)\\)", ""))
  val titleWords = titleFiltered.map(title => title.split(" "))
  val allTerms = titleWords.flatMap(x => x).distinct().zipWithIndex().collectAsMap()

  def getVectors(broadcast: Broadcast[scala.collection.Map[java.lang.String,Long]]): RDD[SparseVector[Double]] = {
    titleWords.map(terms => createVector(terms, broadcast.value))
  }

  def createVector(terms: Array[String], termDict: scala.collection.Map[String, Long]): SparseVector[Double] = {
    val numTerms = termDict.count(x => true)
    val vec = SparseVector.zeros[Double](numTerms)
    for (term <- terms){
      if (termDict.get(term).isDefined){
        vec(termDict.get(term).get.toInt) = 1
      }
    }
    normalize(vec)
  }
}
