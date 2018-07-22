package ru.albemuth.tentura.cs231n.assignment1

import org.slf4j.LoggerFactory
import ru.albemuth.tentura.cs231n.assignment1.DataUtils.{NDArray, array2DataArray, load_CIFAR10, str}
import ru.albemuth.tentura.cs231n.assignment1.classifiers.LinearSVM
import ru.albemuth.tentura.tensor.{MathFunctions, Matrix, Vector}

import scala.util.Random

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
object Svm extends App {

  val LOGGER = LoggerFactory.getLogger(Svm.getClass)

  val (x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data, x_dev_data, y_dev_data) = prepareData()

  val X_dev = Matrix[Float](x_dev_data.shape(0), x_dev_data.shape(1)).of(x_dev_data.data)
  val y_dev = Vector.of[Int](y_dev_data.data)

  val W = Matrix[Float](3073, 10).of(Random.nextGaussian().toFloat) * 0.0001f

  val t1 = System.nanoTime()
  val (loss_vectorized, _) = LinearSVM.svm_loss_vectorized(W, X_dev, y_dev, 0.000005f)
  val t2 = System.nanoTime()
  LOGGER.info(s"First vectorized loss: $loss_vectorized computed in ${t2 - t1}ns")

  val t3 = System.nanoTime()
  val (loss_vectorized2, _) = LinearSVM.svm_loss_vectorized(W, X_dev, y_dev, 0.000005f)
  val t4 = System.nanoTime()
  LOGGER.info(s"Second vectorized loss: $loss_vectorized2 computed in ${t4 - t3}ns")

  def prepareData(): (NDArray[Float], NDArray[Int], NDArray[Float], NDArray[Int], NDArray[Float], NDArray[Int], NDArray[Float], NDArray[Int]) = {

    var (x_train_data, y_train_data, x_test_data, y_test_data) = loadData()

    val num_training = 49000
    val num_validation = 1000
    val num_test = 1000
    val num_dev = 500

    var x_val_data = x_train_data.slice(num_training, num_training + num_validation, 0)
    var y_val_data = y_train_data.slice(num_training, num_training + num_validation, 0)

    x_test_data = x_test_data.slice(0, num_test, 0)
    y_test_data = y_test_data.slice(0, num_test, 0)

    val dev_indices = Random.shuffle(0 to x_train_data.shape(0)).take(num_dev).toArray
    var x_dev_data = x_train_data(dev_indices, 0)
    var y_dev_data = y_train_data(dev_indices, 0)

    x_train_data = x_train_data.slice(0, num_training, 0)
    y_train_data = y_train_data.slice(0, num_training, 0)


    LOGGER.info("")
    LOGGER.info(s"Train data shape: ${str(x_train_data.shape)}")
    LOGGER.info(s"Train labels shape: ${str(y_train_data.shape)}")
    LOGGER.info(s"Validation data shape: ${str(x_val_data.shape)}")
    LOGGER.info(s"Validation labels shape: ${str(y_val_data.shape)}")
    LOGGER.info(s"Test data shape: ${str(x_test_data.shape)}")
    LOGGER.info(s"Test labels shape: ${str(y_test_data.shape)}")
    LOGGER.info(s"Dev data shape: ${str(x_dev_data.shape)}")
    LOGGER.info(s"Dev labels shape: ${str(y_dev_data.shape)}")

    x_train_data = x_train_data.reshape(Array(x_train_data.shape(0), x_train_data.shape.slice(1, x_train_data.shape.length).product))
    x_val_data = x_val_data.reshape(Array(x_val_data.shape(0), x_val_data.shape.slice(1, x_val_data.shape.length).product))
    x_test_data = x_test_data.reshape(Array(x_test_data.shape(0), x_test_data.shape.slice(1, x_test_data.shape.length).product))
    x_dev_data = x_dev_data.reshape(Array(x_dev_data.shape(0), x_dev_data.shape.slice(1, x_dev_data.shape.length).product))
    LOGGER.info("")
    LOGGER.info(s"Reshaped train data shape: ${str(x_train_data.shape)}")
    LOGGER.info(s"Reshaped validation data shape: ${str(x_val_data.shape)}")
    LOGGER.info(s"Reshaped test data shape: ${str(x_test_data.shape)}")
    LOGGER.info(s"Reshaped dev data shape: ${str(x_dev_data.shape)}")

    val mean_image = x_train_data.mean(0)
    LOGGER.info(s"${str(mean_image.data.slice(0, 10))}")

    x_train_data = x_train_data - mean_image
    x_val_data = x_val_data - mean_image
    x_test_data = x_test_data - mean_image
    x_dev_data = x_dev_data - mean_image

    x_train_data = x_train_data.addColumn(1)
    x_val_data = x_val_data.addColumn(1)
    x_test_data = x_test_data.addColumn(1)
    x_dev_data = x_dev_data.addColumn(1)

    LOGGER.info("")
    LOGGER.info(s"Result train data shape: ${str(x_train_data.shape)}")
    LOGGER.info(s"Result validation data shape: ${str(x_val_data.shape)}")
    LOGGER.info(s"Result test data shape: ${str(x_test_data.shape)}")
    LOGGER.info(s"Result dev data shape: ${str(x_dev_data.shape)}")

    (x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data, x_dev_data, y_dev_data)
  }

  def loadData(): (NDArray[Float], NDArray[Int], NDArray[Float], NDArray[Int]) = {
    val cifar10_dir = "src/main/resources/datasets/cifar-10-batches-py"
    LOGGER.info("start loading...")
    val (x_train, y_train, x_test, y_test) = load_CIFAR10(cifar10_dir)
    LOGGER.info("complete")
    LOGGER.info(s"X_train: ${str(x_train.shape())}")
    LOGGER.info(s"y_train: ${str(y_train.shape())}")
    LOGGER.info(s"X_test: ${str(x_test.shape())}")
    LOGGER.info(s"y_test: ${str(y_test.shape())}")

    (NDArray(x_train), NDArray(y_train), NDArray(x_test), NDArray(y_test))

//    val sampled_x_train = subsample(x_train, trainSize)
//    val sampled_y_train = subsample(y_train, trainSize)
//
//    val sampled_x_test = subsample(x_test, testSize)
//    val sampled_y_test = subsample(y_test, testSize)
//
//    val reshaped_x_train = reshape(sampled_x_train)
//    val reshaped_x_test = reshape(sampled_x_test)
//    LOGGER.info("reshaped X_train: {}, {}", reshaped_x_train.length, reshaped_x_train(0).length)
//    LOGGER.info("reshaped X_test: {}, {}", reshaped_x_test.length, reshaped_x_test(0).length)
//
//    (reshaped_x_train, sampled_y_train, reshaped_x_test, sampled_y_test)
  }

}
