import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-27.
  */
class MovieTransform(rawFields: RDD[String]) extends Serializable{
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

  val yearsFilled = years.map(x => if (x == 1900) yearsMean else x)





  def getAgeAggregated: Array[(Double, Long)] = {
    yearsFilled.map(x => 1998 - x).countByValue().toArray.sortBy(_._1)
  }
}
