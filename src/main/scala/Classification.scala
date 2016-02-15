import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD, LogisticRegressionWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}


/**
  * Created by vishalkuo on 2016-02-13.
  */
object Classification {
  val conf = new SparkConf().setMaster("local[2]")
    .setAppName("StumbleUpon Classifier").set("spark.executor.memory", "2g")
  val iterations = 10
  val treeDepth = 5
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
  }

  def logisticRegression(data: RDD[LabeledPoint])= {
    val model = LogisticRegressionWithSGD.train(data,iterations)
    val correctCount = data.map(lPoint =>
    if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
  }

  def svm(data: RDD[LabeledPoint]) = {
    val model = SVMWithSGD.train(data, iterations)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
  }

  def naiveBayes(data: RDD[LabeledPoint]) = {
    val model = NaiveBayes.train(data)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
  }

  def decisionTree(data: RDD[LabeledPoint]) = {
    val model = DecisionTree.train(data, Algo.Classification, Entropy, treeDepth)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
  }

}
