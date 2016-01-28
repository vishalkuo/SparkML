import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-27.
  */
class MovieTransform(sc: SparkContext, rawFields: RDD[String]) {
  val movieFields = rawFields.map(lines => lines.split("\\|"))
  val years = movieFields.map(fields => fields(2)).map({
    field => try{
      field.substring(field.length - 4).toInt
    } catch {
      case _: NumberFormatException | _: StringIndexOutOfBoundsException => 1900
    }
  })

  val yearsFiltered = years.filter(_ != 1900)
  val yearsMean = yearsFiltered.mean()
//
//  def getYearsFilled: RDD[Int] = {
//    years.map({
//      case 1900 => yearsMean.toInt
//      case x => x.toInt
//    })
//  }
//
//
//
//  def getAgeAggregated: Array[(Int, Long)] = {
//    getYearsFilled.map(x => 1998 - x).countByValue().toArray.sortBy(_._1)
//  }
}
