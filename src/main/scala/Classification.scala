import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import shapeless.record


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
      val positiveFeature = feature.map(x => if (x < 0) 0 else x)
      LabeledPoint(label, Vectors.dense(positiveFeature))
    }
    mllibData.cache()
    naiveBayesData.cache()

//    naiveBayes(naiveBayesData)
//    val logReg = logisticRegression(mllibData)
//    val svmModel = svm(mllibData)
//    val comp = lrSVMComparison(logReg, svmModel, mllibData)
//    val nb = naiveBayes(naiveBayesData)
//    naiveBayesMetrics(nb, naiveBayesData)

    val catMap = fields.map(field => field(3))
      .distinct()
      .collect()
      .zipWithIndex
      .toMap

    val numCategories = catMap.size

    val dataWithCategories  =fields.map({case record =>
      val cleansed = record.map(_.replaceAll("\"", ""))
      val label = cleansed(record.length - 1).toInt
      val category = catMap(record(3))
      val catVector = Array.ofDim[Double](numCategories)
      catVector(category) = 1
      val baseFeatures = cleansed.slice(4, record.length - 1).map(item => if (item.equals("?")) 0.0 else item.toDouble)
      val allFeatures = catVector ++ baseFeatures
      LabeledPoint(label, Vectors.dense(allFeatures))
    })

    val categoryScaler = new StandardScaler(withMean = true, withStd = true)
      .fit(dataWithCategories.map(p => p.features))
    val scaledCategorized = dataWithCategories.map(p =>
      LabeledPoint(p.label, categoryScaler.transform(p.features))
    )
    logisticRegression(scaledCategorized)

  }

  def logisticRegression(data: RDD[LabeledPoint]): classification.LogisticRegressionModel= {
    val model = LogisticRegressionWithSGD.train(data,iterations)
    val correctCount = data.map(lPoint =>
    if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(s"sum: $correctCount, acc: $acc")
    model
  }

  def logRegWithScaling(mllibData: RDD[LabeledPoint]) = {
    val vectors = mllibData.map(labeledPoint => labeledPoint.features)
    val matrix = new RowMatrix(vectors)
    val summary = matrix.computeColumnSummaryStatistics()

    val vectorScaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = mllibData.map(labeledPoint => LabeledPoint(labeledPoint.label,
      vectorScaler.transform(labeledPoint.features)))

    val logRegScaledModel = logisticRegression(scaledData)

    val comparison = scaledData.map(point =>
      (logRegScaledModel.predict(point.features), point.label))
    val metrics = new BinaryClassificationMetrics(comparison)
    val lrPR = metrics.areaUnderPR()
    val lrRoc = metrics.areaUnderROC()
    println(s"Area under PR: $lrPR, area under ROC: $lrRoc")
  }
  def svm(data: RDD[LabeledPoint]): SVMModel = {
    val model = SVMWithSGD.train(data, iterations)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
    model
  }

  def naiveBayes(data: RDD[LabeledPoint]): NaiveBayesModel = {
    val model = NaiveBayes.train(data)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
    model
  }

  def decisionTree(data: RDD[LabeledPoint]): DecisionTreeModel = {
    val model = DecisionTree.train(data, Algo.Classification, Entropy, treeDepth)
    val correctCount = data.map(lPoint =>
      if (model.predict(lPoint.features) == lPoint.label) 1 else 0).sum
    val acc = correctCount / data.count()
    println(acc)
    model
  }

  def lrSVMComparison(logReg: classification.LogisticRegressionModel, svmModel: SVMModel, mllibData: RDD[LabeledPoint]):
  Seq[(String, Double, Double)] = {
    val comparison = Seq(logReg, svmModel).map{case model =>
      val accAndCategory = mllibData.map{case point =>
        (model.predict(point.features),  point.label)
      }
      val metric = new BinaryClassificationMetrics(accAndCategory)
      (model.getClass.getSimpleName, metric.areaUnderPR(), metric.areaUnderROC())
    }
    comparison
  }

  def naiveBayesMetrics(nb: NaiveBayesModel, nbData: RDD[LabeledPoint]):Seq[(String, Double, Double)] = {
    val comparison = Seq(nb).map({case model =>
        val accAndCategory = nbData.map({ case point =>
          (model.predict(point.features), point.label)
        })
        val met = new BinaryClassificationMetrics(accAndCategory)
      (model.getClass.getSimpleName, met.areaUnderPR(), met.areaUnderROC())
    })
    comparison
  }

  def dTreeMetrics(dtModel: DecisionTreeModel, mllibData: RDD[LabeledPoint]): Seq[(String, Double, Double)] = {
    val comparison = Seq(dtModel).map({case model =>
        val accAndCategory = mllibData.map{case point =>
          val res = model.predict(point.features)
          (if (res > 0.5) 1.0 else 0, point.label)
        }
      val met = new BinaryClassificationMetrics(accAndCategory)
      (model.getClass.getSimpleName, met.areaUnderPR(), met.areaUnderROC())
    })
    comparison
  }

}
