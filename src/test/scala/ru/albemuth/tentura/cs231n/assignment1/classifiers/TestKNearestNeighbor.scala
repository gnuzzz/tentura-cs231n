package ru.albemuth.tentura.cs231n.assignment1.classifiers

import org.scalatest.FunSuite
import ru.albemuth.tentura.ResultsCache
import ru.albemuth.tentura.tensor.Matrix.matrixRow
import ru.albemuth.tentura.tensor.VectorFunctions.{argmax, bincount}
import ru.albemuth.tentura.tensor.kernel.vector.VectorKernel
import ru.albemuth.tentura.tensor.{Matrix, Vector, VectorFunctions}
import ru.albemuth.tentura.util.Memory

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
class TestKNearestNeighbor extends FunSuite {

  test("results cache") {
    val distances = new Matrix[Float](1000, 4000)
    println(distances.result(matrixRow.kernel[Float], 0, vector(4000)))
    println(distances.result(matrixRow.kernel[Float], 0, vector(4000)))
//    println(distances.result(matrixRow, 0, vector(4000)))
//    println(distances.result(matrixRow, 0, vector(4000)))
    println(VectorKernel.vector(matrixRow, distances, 0, vector(distances.columns)))
    println(VectorKernel.vector(matrixRow, distances, 0, vector(distances.columns)))
    println(VectorKernel.vector(matrixRow, distances, 0, vector(distances.columns)))
    println(VectorKernel.vector(matrixRow, distances, 0, vector(distances.columns)))
  }

  test("Memory leak") {
    val resultsCache = new ResultsCache
    val yTrain = new Vector[Int](4000)
    val distances = new Matrix[Float](1000, 4000)
    val k = 4
    val yPred = Memory.check("111"){new Vector[Int](distances.rows)}
    val row = new Vector[Float](4000)
    val k_choices = Array(1, 3, 5, 8, 10, 12, 15, 20, 50, 100)
    val num_folds = 5
    for (k <- k_choices) {
      println(s"$k: ")
      val accuracies = for (i <- 0 until num_folds) yield {
        loop(distances, k)
      }
    }
  }

  val resultsCache = new ResultsCache
  val yTrain = new Vector[Int](4000)

  def loop(distances: Matrix[Float], k: Int): Vector[Float] = {
    val yPred = Memory.check("yPred"){new Vector[Int](distances.rows)}
    val row = Memory.check("row"){new Vector[Float](distances.row(0).length)}
    val argsort = Memory.check("argsort"){resultsCache.result(this, distances.row(0).length, VectorFunctions.argsort(distances.row(0)))}
    for (i <- 0 until distances.rows) {
      //      val closestY = yTrain.values(argsort(distances.row(i)).slice(0, k))
      //      argmax(bincount(closestY), yPred(i)) //yPred(i) = VectorFunctions.bincount(closestY).argmax().value()
//      val a = Memory.check(s"distances.row(i): $i"){distances.row(i)}
      val a: Vector[Float] = Memory.check(s"distances.row(i, row): $i"){distances.row(i, row)}
      //      val a = Memory.check(s"1: $i"){distances.row(i, row)}
      val b: Vector[Int] = Memory.check("argsort(a)"){VectorFunctions.argsort(a)}
//      val c = Memory.check("3"){b.slice(0, k)}
      val c = Memory.check("b(0, k)"){b(0, k)}
      val closestY = Memory.check("yTrain.values(c)"){yTrain.values(c)}
      Memory.check("argmax(bincount(closestY), yPred(i))"){argmax(bincount(closestY), yPred(i))} //yPred(i) = VectorFunctions.bincount(closestY).argmax().value()
    }
    null
  }

  def vector(length: Int): Vector[Float] = {
    new Vector[Float](length)
  }

}
