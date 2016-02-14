import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by vishalkuo on 2016-02-13.
  */
object Classification {
  val conf = new SparkConf().setMaster("local[2]")
    .setAppName("StumbleUpon Classifier").set("spark.executor.memory", "2g")
  val sc = new SparkContext(conf)
  def main(args: Array[String]) {
    val rawData = sc.textFile("src/main/resources/datasets/StumbleUpon/train.tsv")
    val fields = rawData.map(lines => lines.split("\t"))
    val cleansedData = fields.map { field =>
      val cleansed = field.map(_.replace("\"", ""))
      val classifiedVal = cleansed(field.length - 1).toInt
      val features = cleansed.slice(4, field.length - 1).map(item => if (item.equals("?")) 0.0 else item.toDouble)
      (classifiedVal, features)
    }

    val mllibData = cleansedData.map { case (classifiedVal, features) =>
      LabeledPoint(classifiedVal, Vectors.dense(features))
    }
    val naiveBayesData = cleansedData.map { case (label, feature) =>
      feature.map(x => if (x < 0) 0 else x)
      LabeledPoint(label, Vectors.dense(feature))
    }
    mllibData.cache()
    println(mllibData.count())
    println(naiveBayesData.count())
  }
}
