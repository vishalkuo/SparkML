import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
  * Created by vishalkuo on 2016-01-26.
  */
class UserTransform(sc: SparkContext, rawFields: RDD[String]) {
  val userFields = rawFields.map(line => line.split("\\|"))

}
