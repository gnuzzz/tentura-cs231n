package ru.albemuth.tentura.cs231n.assignment1

import scala.reflect.ClassTag

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
class Folds[X: ClassTag, Y: ClassTag](x: Array[Array[X]], y: Array[Y], numFolds: Int) {

  private val xFolds: Array[Array[Array[X]]] = Folds.split(x, numFolds)
  private val yFolds: Array[Array[Y]] = Folds.split(y, numFolds)
  val valSize: Int = x.length / numFolds
  val trainSize: Int = x.length - valSize

  def cvData(validationFoldIndex: Int): (Array[Array[X]], Array[Y], Array[Array[X]], Array[Y]) = {
    val (xTrain, xVal) = Folds.cvData[Array[X]](xFolds, validationFoldIndex)
    val (yTrain, yVal) = Folds.cvData[Y](yFolds, validationFoldIndex)
    (xTrain, yTrain, xVal, yVal)
  }
}

object Folds {

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

}