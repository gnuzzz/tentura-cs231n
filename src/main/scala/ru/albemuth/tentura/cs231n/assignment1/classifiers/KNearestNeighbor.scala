package ru.albemuth.tentura.cs231n.assignment1.classifiers

import jcuda.jcublas.{cublasHandle, cublasOperation}
import ru.albemuth.jcuda.jcusegsort.KeyValueSortContext
import ru.albemuth.tentura.ResultsCache
import ru.albemuth.tentura.tensor.{Matrix, MatrixFunctions, Vector, VectorFunctions}
import ru.albemuth.tentura.tensor.Scalar._

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
class KNearestNeighbor {

  private var xTrain: Matrix[Float] = _
  private var yTrain: Vector[Int] = _
  private val resultsCache = new ResultsCache

  def train(x: Matrix[Float], y: Vector[Int]): Unit = {
    xTrain = x
    yTrain = y
  }

  def predict(x: Matrix[Float], k: Int = 1, numLoops: Int = 0, context: KeyValueSortContext)(implicit handle: cublasHandle): Vector[Int] = {
    val distances = numLoops match {
      case 0 => computeDistancesNoLoops(x)
      case _ => throw new IllegalArgumentException(s"Invalid value $numLoops for numLoops")
    }
    predictLabels(distances, k, context)
  }

  def computeDistancesNoLoops(x: Matrix[Float])(implicit handle: cublasHandle): Matrix[Float] = {
    //(X ** 2).sum(axis=1)[:, None] - 2 * X.dot(self.X_train.T) + (self.X_train ** 2).sum(axis=1)
//    (x ^ 2).sum(axis=1) -| (2 * x * xTrain.T) + (xTrain ^ 2).sum(axis=1)
    val c = MatrixFunctions.gemm(2, x, cublasOperation.CUBLAS_OP_N, xTrain, cublasOperation.CUBLAS_OP_T, 0, x.result("gemm", xTrain, new Matrix[Float](x.rows, xTrain.rows)))
    (x ^ 2).sum(axis=1) -| c + (xTrain ^ 2).sum(axis=1)
  }

  def predictLabels(distances: Matrix[Float], k: Int, context: KeyValueSortContext): Vector[Int] = {
//    num_test = dists.shape[0]
//    y_pred = np.zeros(num_test)
//    for i in xrange(num_test):
//      closest_y = []
//      closest_y = self.y_train[np.argsort(dists[i])[:k]]
//      y_pred[i] = np.bincount(closest_y).argmax()
//    y_pred

    //01:13
//    val yPred = new Vector[Int](distances.rows)
//    val row = new Vector[Float](distances.row(0).length)
//    for (i <- 0 until distances.rows) {
//      val closestY = yTrain.values(VectorFunctions.argsort(distances.row(i, row), context).slice(0, k))
//      VectorFunctions.argmax(VectorFunctions.bincount(closestY), yPred(i)) //yPred(i) = VectorFunctions.bincount(closestY).argmax().value()
//    }

    //00:03,438 kernel
    //00:01,631 cuBLAS
    //00:00,960 cuBLAS without copy folds
    val closestY = yTrain.values(MatrixFunctions.argsort(distances, context).slice(0, k, axis = 1))
    val yPred = MatrixFunctions.argmax(MatrixFunctions.bincount(closestY, axis = 1), axis = 1)

    yPred
  }

}
