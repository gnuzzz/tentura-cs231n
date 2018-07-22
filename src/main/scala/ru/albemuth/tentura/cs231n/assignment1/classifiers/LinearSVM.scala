package ru.albemuth.tentura.cs231n.assignment1.classifiers

import ru.albemuth.tentura.tensor.{Axis, Comparator, MathFunctions, Matrix, MatrixCasFunctions, Vector}
import ru.albemuth.tentura.tensor.Scalar._

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
object LinearSVM {

  def svm_loss_vectorized(W: Matrix[Float], X: Matrix[Float], y: Vector[Int], reg: Float): (Float, Matrix[Float]) = {
//    num_train = X.shape[0]
//    scores = X.dot(W)
//    correct_class_scores = scores[np.arange(num_train), y]
//    margins = np.maximum(0, scores - np.array([correct_class_scores]).T + 1)
//    margins[np.arange(num_train), y] = 0
//    loss += np.mean(np.sum(margins, axis=1))
//    loss += reg * np.sum(W * W)
//    binary = margins
//    binary[margins > 0] = 1
//    mc = np.sum(binary, axis=1)
//    binary[np.arange(len(y)), y] -= mc
//    dW = X.T.dot(binary)
//    dW /= num_train
//    dW += reg * 2 * W

    val num_train = X.rows
    val scores = X * W
    val correct_class_scores = scores(y, axis = 0)
    val margins = MathFunctions.max[Float](scores -| correct_class_scores + 1.0f, 0)
    margins(y, axis = 0) = 0
//    for (v <- margins.sum(axis = 1).values()) print(s"$v ")
//    println()
//    println(margins.sum(axis = 1).mean().value())
//    println((W :* W).sum().value())
    val loss = margins.sum(axis = 1).mean() + reg * (W :* W).sum()

    val binary = MatrixCasFunctions.cas[Float](margins, Comparator.>, 0, 1)
    val mc = binary.sum(axis = 1)
    binary(y, axis = 0) -= mc
    val dW = X.T * binary / num_train + reg * 2.0f * W
    (loss, dW)
  }

}
