package ru.albemuth.tentura.cs231n.assignment1

import org.slf4j.LoggerFactory
import ru.albemuth.jcuda.jcusegsort.{Datatype, KeyValueSortContext, Sorting}
import ru.albemuth.tentura.cs231n.assignment1.classifiers.KNearestNeighbor
import ru.albemuth.tentura.tensor.{Matrix, Vector}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
object Knn extends App {

  val LOGGER = LoggerFactory.getLogger(Knn.getClass)

  val train_size = 5000
  val test_size = 500

  val (x_train_data, y_train_data, x_test_data, y_test_data) = loadData(train_size, test_size)

  val X_train = Matrix.of(x_train_data)
  val y_train = Vector.of(y_train_data)
  val X_test = Matrix.of(x_test_data)
  val y_test = Vector.of(y_test_data)
  val featuresNum = X_train.columns

  val classifier = new KNearestNeighbor()
  classifier.train(X_train, y_train)

  val t1_1 = System.nanoTime()
  classifier.computeDistancesNoLoops(X_test)
  val t2_1 = System.nanoTime()
  LOGGER.info("computeDistancesNoLoops first: {}", t2_1 - t1_1)
  val t1_2 = System.nanoTime()
  val distances = classifier.computeDistancesNoLoops(X_test)
  val t2_2 = System.nanoTime()
  LOGGER.info("computeDistancesNoLoops second: {}", t2_2 - t1_2)
//  val context = Sorting.keyValueSortContext(Datatype.FLOAT, Datatype.INT, distances.columns, 1)
  val context = Sorting.keyValueSortContext(Datatype.FLOAT, Datatype.INT, distances.rows * distances.columns, distances.rows)
  val y_pred_1 = classifier.predictLabels(distances, k = 1, context)
  val correct_1 = similarity(y_pred_1.values(), y_test_data)
  val accuracy_1 = correct_1.toFloat / test_size
  LOGGER.info("Got {} / {} correct => accuracy: {}", correct_1.asInstanceOf[Object], test_size.asInstanceOf[Object], accuracy_1.asInstanceOf[Object])

//  val y_pred_5 = classifier.predictLabels(distances, k = 5, context)
  val y_pred_5 = classifier.predictLabels(classifier.computeDistancesNoLoops(X_test), k = 5, context)
  val correct_5 = similarity(y_pred_5.values(), y_test_data)
  val accuracy_5 = correct_5.toFloat / test_size
  LOGGER.info("Got {} / {} correct => accuracy: {}", correct_5.asInstanceOf[Object], test_size.asInstanceOf[Object], accuracy_5.asInstanceOf[Object])

  val num_folds = 5
  val k_choices = Array(1, 3, 5, 8, 10, 12, 15, 20, 50, 100)

  val folds = new Folds(x_train_data, y_train_data, num_folds)
  val X_tr = new Matrix[Float](folds.trainSize, featuresNum)
  val y_tr = new Vector[Int](folds.trainSize)
  val X_val = new Matrix[Float](folds.valSize, featuresNum)
  val y_val = new Vector[Int](folds.valSize)

  val k2accuracies = new mutable.HashMap[Int, Seq[Float]]()
//  val valContext = Sorting.keyValueSortContext(Datatype.FLOAT, Datatype.INT, 4000, 1)
  val valContext = Sorting.keyValueSortContext(Datatype.FLOAT, Datatype.INT, 1000 * 4000, 1000)

  val X_tr_folds = Array.ofDim[Array[Float]](num_folds)
  val X_val_folds = Array.ofDim[Array[Float]](num_folds)
  val y_tr_folds = Array.ofDim[Array[Int]](num_folds)
  val y_val_folds = Array.ofDim[Array[Int]](num_folds)
  for (i <- 0 until num_folds) {
    val (xTrData, yTrData, xValData, yValData) = folds.cvData(i)
    X_tr_folds(i) = Matrix.data(X_tr.rows, X_tr.columns)(xTrData)
    y_tr_folds(i) = yTrData
    X_val_folds(i) = Matrix.data(X_val.rows, X_val.columns)(xValData)
    y_val_folds(i) = yValData
  }

  classifier.train(X_tr, y_tr)
  val dists = classifier.computeDistancesNoLoops(X_val)

  for (k <- k_choices) {
    val accuracies = for (i <- 0 until num_folds) yield {
//      val (xTrData, yTrData, xValData, yValData) = folds.cvData(i)
//      X_tr.copy2device(Matrix.data(X_tr.rows, X_tr.columns)(xTrData))
//      y_tr.copy2device(yTrData)
//      X_val.copy2device(Matrix.data(X_val.rows, X_val.columns)(xValData))
//      classifier.train(X_tr, y_tr)
//      val y_pred = classifier.predict(X_val, k, 0, valContext)
//      val num_correct = similarity(y_pred.values(), yValData)
//      num_correct / yValData.length.toFloat


      X_tr.copy2device(X_tr_folds(i))
      y_tr.copy2device(y_tr_folds(i))
      X_val.copy2device(X_val_folds(i))
      classifier.train(X_tr, y_tr)
      val y_pred = classifier.predict(X_val, k, 0, valContext)
      val num_correct = similarity(y_pred.values(), y_val_folds(i))
      num_correct / folds.valSize.toFloat
    }
    k2accuracies(k) = accuracies
  }

  val ks = for (k <- k_choices) yield {
    val accuracies = k2accuracies(k)
    LOGGER.info("k = {}, accuracy = {}, mean = {}", k.asInstanceOf[Object], accuracies.mkString(", ").asInstanceOf[Object], (accuracies.sum / accuracies.length).asInstanceOf[Object])
    (k, accuracies.sum / accuracies.length)
  }

  val best_k = ks.maxBy(_._2)._1
  val bestClassifier = new KNearestNeighbor
  bestClassifier.train(X_train, y_train)
  val bestDistances = bestClassifier.computeDistancesNoLoops(X_test)
  val bestContext = Sorting.keyValueSortContext(Datatype.FLOAT, Datatype.INT, bestDistances.rows * bestDistances.columns, bestDistances.rows)
  val y_test_pred = bestClassifier.predictLabels(bestDistances, k = best_k, bestContext)
  val num_correct = similarity(y_test_pred.values(), y_test_data)
  val best_accuracy = num_correct / y_test_data.length.toFloat
  LOGGER.info("Best k = {}: got {} / {} correct => accuracy: {}", best_k.asInstanceOf[Object], num_correct.asInstanceOf[Object], y_test_data.length.asInstanceOf[Object], best_accuracy.asInstanceOf[Object])

  def loadData(trainSize: Int, testSize: Int): (Array[Array[Float]], Array[Int], Array[Array[Float]], Array[Int]) = {
    val cifar10_dir = "src/main/resources/datasets/cifar-10-batches-py"
    LOGGER.info("start loading...")
    val (x_train, y_train, x_test, y_test) = DataUtils.load_CIFAR10(cifar10_dir)
    LOGGER.info("complete")
    LOGGER.info("X_train: {}, {}, {}, {}", x_train.length.asInstanceOf[Object], x_train(0).length.asInstanceOf[Object], x_train(0)(0).length.asInstanceOf[Object], x_train(0)(0)(0).length.asInstanceOf[Object])
    LOGGER.info("y_train: {}", y_train.length)
    LOGGER.info("X_test: {}, {}, {}, {}", x_test.length.asInstanceOf[Object], x_test(0).length.asInstanceOf[Object], x_test(0)(0).length.asInstanceOf[Object], x_test(0)(0)(0).length.asInstanceOf[Object])
    LOGGER.info("y_test: {}", y_test.length)

    val sampled_x_train = subsample(x_train, trainSize)
    val sampled_y_train = subsample(y_train, trainSize)

    val sampled_x_test = subsample(x_test, testSize)
    val sampled_y_test = subsample(y_test, testSize)

    val reshaped_x_train = reshape(sampled_x_train)
    val reshaped_x_test = reshape(sampled_x_test)
    LOGGER.info("reshaped X_train: {}, {}", reshaped_x_train.length, reshaped_x_train(0).length)
    LOGGER.info("reshaped X_test: {}, {}", reshaped_x_test.length, reshaped_x_test(0).length)

    (reshaped_x_train, sampled_y_train, reshaped_x_test, sampled_y_test)
  }

  def subsample[T: ClassTag](data: Array[T], num_samples: Int): Array[T] = {
    val ret = new Array[T](num_samples)
    System.arraycopy(data, 0, ret, 0, num_samples)
    ret
  }

  def reshape[T: ClassTag](data: Array[Array[Array[Array[T]]]]): Array[Array[T]] = {
    val samplesNum = data.length
    val rowsNum = data(0).length
    val columnsNum = data(0)(0).length
    val channelsNum = data(0)(0)(0).length
    val sampleLength = rowsNum * columnsNum * channelsNum
    val reshapedData: Array[Array[T]] = Array.ofDim(data.length, sampleLength)
    for (i <- data.indices) {
      val sample = data(i)
      val reshapedSample = reshapedData(i)
      for (j <- sample.indices) {
        val row = sample(j)
        for (k <- row.indices) {
          val column = row(k)
          for (l <- column.indices) {
            val reshapedIndex = (j * columnsNum + k) * channelsNum + l
            reshapedSample(reshapedIndex) = column(l)
          }
        }
      }
    }
    reshapedData
  }

  def split[T: ClassTag](data: Array[T], numFolds: Int): Array[Array[T]] = {
    val folds: Array[Array[T]] = Array.ofDim(numFolds, data.length / numFolds)
    for (i <- folds.indices) {
      val fold = folds(i)
      System.arraycopy(data, fold.length * i, fold, 0, fold.length)
    }
    folds
  }

  def cvData[T: ClassTag](folds: Array[Array[T]], validationFoldIndex: Int): (Array[T], Array[T]) = {
    val trainData = (for (i <- folds.indices if i != validationFoldIndex) yield folds(i)).flatten.toArray
    val validationData = folds(validationFoldIndex)
    (trainData, validationData)
  }

  def similarity(y1: Array[Int], y2: Array[Int]): Int = {
    if (y1.length != y2.length) throw new IllegalArgumentException(s"${y1.length} != ${y2.length}")
    y1.zip(y2).map({case (v1, v2) => v1 - v2}).count(v => v == 0)
  }

  def similarity(data1: Array[Array[Int]], data2: Array[Array[Int]]): Int = {
    if (data1.length != data2.length) throw new IllegalArgumentException(s"${data1.length} != ${data2.length}")
    data1.zip(data2).map({case (v1, v2) => similarity(v1, v2)}).sum
  }

}
