package ru.albemuth.tentura.cs231n.assignment1

import java.io.{File, FileInputStream}

import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import ru.albemuth.tentura.cs231n.TryWith

import scala.reflect.ClassTag

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
object DataUtils {

  def load_CIFAR10(root: String): (Array[Array[Array[Array[Float]]]], Array[Int], Array[Array[Array[Array[Float]]]], Array[Int]) = {
    val x_train = load(new File(root, "X_train.json"), new TypeReference[Array[Array[Array[Array[Float]]]]]() {})
    val y_train = load(new File(root, "y_train.json"), new TypeReference[Array[Int]]() {})
    val x_test = load(new File(root, "X_test.json"), new TypeReference[Array[Array[Array[Array[Float]]]]]() {})
    val y_test = load(new File(root, "y_test.json"), new TypeReference[Array[Int]]() {})

    (x_train, y_train, x_test, y_test)
  }

  def load[T: ClassTag](file: File, typeReference: TypeReference[T]): T = {
    TryWith(new FileInputStream(file)) (in => {
      val mapper = new ObjectMapper
      mapper.readValue(in, typeReference).asInstanceOf[T]
    })
  }

  def str[T](array: Array[T]): String = {
    val sb = new StringBuilder
    sb.append("(")
    for (item <- array) {
      if (sb.size > 1) sb.append(", ")
      item match {
        case items: Array[_] =>
          sb.append(str(items))
        case _ =>
          sb.append(item.toString)
      }
    }
    sb.append(")")
    sb.toString()
  }

  implicit def array2DataArray[T: ClassTag](array: Array[T]): DataArray[T] = new DataArray[T](array)

  class DataArray[T: ClassTag](val array: Array[T]) {

    def deepClone(): Array[T] = {
      DataArray.deepClone(array)
    }

    def get[ITEM: ClassTag](index: Array[Int]): ITEM = {
      DataArray.get[ITEM](array, index)
    }

    def set[ITEM: ClassTag](index: Array[Int], value: ITEM): Unit = {
      DataArray.set[ITEM](array, index, value)
    }

    def shape(): Array[Int] = {
      def shape(item: Any, sh: List[Int]): List[Int] = {
        item match {
          case items: Array[_] =>
            if (items.length == 0) {
              sh
            } else {
              shape(items(0), sh :+ items.length)
            }
          case _ =>
            sh
        }
      }
      shape(array, List()).toArray
    }

    def nextIndex(shape: Array[Int], index: Array[Int]): Array[Int] = {
      def nextIndex(shape: Array[Int], dim: Int, next: Array[Int]): Array[Int] = {
        val idx = next(dim) + 1
        if (idx < shape(dim)) {
          next(dim) = idx
          next
        } else if (dim == 0) {
          throw new IndexOutOfBoundsException(s"Can't create next index for index ${str(index)} and shape ${str(shape)}")
        } else {
          next(dim) = 0
          nextIndex(shape, dim - 1, next)
        }
      }
      if (shape.length != index.length) throw new IllegalArgumentException(s"Index ${str(index)} does not match shape ${str(shape)}")
      nextIndex(shape, shape.length - 1, index.clone())
    }

    def nextIndex(index: Array[Int]): Array[Int] = {
      nextIndex(shape(), index)
    }

    def reshape[ITEM: ClassTag](newShape: Array[Int]): Array[_] = {
      def fill(values: Array[_], shape: Array[Int], index: Array[Int], reshaped: Array[_], reshapedShape: Array[Int], reshapedIndex: Array[Int], itemIndex: Int, itemsNumber: Int): Unit = {
        val value = DataArray.get[ITEM](values, index)
        DataArray.set[ITEM](reshaped, reshapedIndex, value)
        if (itemIndex < itemsNumber - 1) {
          fill(values, shape, nextIndex(shape, index), reshaped, reshapedShape, nextIndex(reshapedShape, reshapedIndex), itemIndex + 1, itemsNumber)
        }
      }

      val currentShape = shape()
      val itemsNumber = currentShape.product
      if (newShape.product != itemsNumber) throw new IllegalArgumentException(s"New shape ${str(newShape)} does not match current shape ${str(currentShape)}")
      val reshaped = DataArray.ofDim[ITEM](newShape)
      fill(array, currentShape, Array.ofDim[Int](currentShape.length), reshaped, newShape, Array.ofDim[Int](newShape.length), 0, itemsNumber)
      reshaped
    }

  }

  object DataArray {

    def deepClone[T: ClassTag](array: Array[T]): Array[T] = {
      val cloned = array.clone()
      for (i <- array.indices) {
        val item = array(i)
        item match {
          case items: Array[_] =>
            cloned(i) = deepClone(items)(ClassTag(items.getClass.getComponentType)).asInstanceOf[T]
          case _ => //do nothing
        }
      }
      cloned
    }

    def ofDim[T: ClassTag](shape: Array[Int]): Array[_] = {
      def array(shape: Array[Int], dim: Int): Array[_] = {
        if (dim == shape.length - 1) {
          Array.ofDim[T](shape(dim))
        } else {
          val firstItem = array(shape, dim + 1)
          Array.fill(shape(dim))(deepClone(firstItem))(ClassTag(firstItem.getClass))
        }
      }
      array(shape, 0)
    }

    def get[ITEM: ClassTag](array: Array[_], index: Array[Int]): ITEM = {
      def get(items: Array[_], index: Array[Int], dim: Int): ITEM = {
        if (dim == index.length - 1) {
          items.asInstanceOf[Array[ITEM]](index(dim))
        } else {
          get(items(index(dim)).asInstanceOf[Array[_]], index, dim + 1)
        }
      }
      get(array, index, 0)
    }

    def set[ITEM: ClassTag](array: Array[_], index: Array[Int], value: ITEM): Unit = {
      def set(items: Array[_], index: Array[Int], dim: Int, value: ITEM): Unit = {
        if (dim == index.length - 1) {
          items.asInstanceOf[Array[ITEM]](index(dim)) = value
        } else {
          set(items(index(dim)).asInstanceOf[Array[_]], index, dim + 1, value)
        }
      }
      set(array, index, 0, value)
    }

  }

  class NDArray[T: ClassTag](val data: Array[T], val shape: Array[Int]) {

    def apply(indices: Array[Int], dim: Int): NDArray[T] = {
      if (dim == 0) {
        val rowLength = shape.slice(1, shape.length).product
        val selectedData = Array.ofDim[T](indices.length * rowLength)
        for (i <- indices.indices) {
          System.arraycopy(data, indices(i) * rowLength, selectedData, i * rowLength, rowLength)
        }
        NDArray(selectedData, (List(indices.length) ++ shape.slice(1, shape.length)).toArray)
      } else {
        ???
      }
    }

    def slice(from: Int, to: Int, dim: Int): NDArray[T] = {
      if (dim == 0) {
        val rowLength = shape.slice(1, shape.length).product
        new NDArray[T](data.slice(from * rowLength, to * rowLength), (List(to - from) ++ shape.slice(1, shape.length)).toArray)
      } else {
        ???
      }
    }

    def reshape(newShape: Array[Int]): NDArray[T] = {
      if (newShape.product != shape.product) throw new IllegalArgumentException(s"New shape ${str(newShape)} does not match current shape ${str(shape)}")
      NDArray(data, newShape)
    }

    def mean(dim: Int): NDArray[Float] = {
      if (dim == 0) {
        if (shape.length == 2) {
          val result = NDArray[Float](Array.ofDim[Float](shape(1)), shape.slice(1, shape.length))
          val clazz = implicitly[ClassTag[T]].runtimeClass
          clazz match {
            case b if b == classOf[Boolean] => ???
            case b if b == classOf[Byte] => ???
            case c if c == classOf[Char] => ???
            case s if s == classOf[Short] => ???
            case i if i == classOf[Int] =>
              for (
                i <- 0 until shape (0);
                j <- 0 until shape (1)
              ) {
                val value = data (i * shape (1) + j).asInstanceOf[Int]
                val resultValue = result.data(j)
                result.data (j) = (value + i * resultValue) / (i + 1).asInstanceOf[Float]
              }
            case l if l == classOf[Long] => ???
            case f if f == classOf[Float] =>
              for (
                i <- 0 until shape (0);
                j <- 0 until shape (1)
              ) {
                val value = data (i * shape (1) + j).asInstanceOf[Float]
                val resultValue = result.data (j)
                result.data (j) = (value + i * resultValue) / (i + 1)
              }
            case d if d == classOf[Double] => ???
            case _ => ??? //not supported
          }
          result
        } else {
          ???
        }
      } else {
        ???
      }
    }

    def -(array: NDArray[T]): NDArray[T] = {
      val result = NDArray[T](data.deepClone(), shape)
      if (shape.length == 2) {
        val clazz = implicitly[ClassTag[T]].runtimeClass
        clazz match {
          case b if b == classOf[Boolean] => ???
          case b if b == classOf[Byte] => ???
          case c if c == classOf[Char] => ???
          case s if s == classOf[Short] => ???
          case i if i == classOf[Int] =>
            for (
              i <- 0 until shape (0);
              j <- 0 until shape (1)
            ) {
              val value = data (i * shape (1) + j).asInstanceOf[Int]
              val r = result.data(j)
              result.asInstanceOf[NDArray[Int]].data(i * shape (1) + j) = value - array.data(j).asInstanceOf[Int]
            }
            result
          case l if l == classOf[Long] => ???
          case f if f == classOf[Float] =>
            for (
              i <- 0 until shape (0);
              j <- 0 until shape (1)
            ) {
              val value = data(i * shape (1) + j).asInstanceOf[Float]
              result.asInstanceOf[NDArray[Float]].data(i * shape (1) + j) = value - array.data(j).asInstanceOf[Float]
            }
            result
          case d if d == classOf[Double] => ???
          case _ => ??? //not supported
        }
      } else {
        ???
      }
    }

    def addColumn(value: => T): NDArray[T] = {
      if (shape.length == 2) {
        val resultData = Array.ofDim[T](shape(0) * (shape(1) + 1))
        val resultShape = Array(shape(0), shape(1) + 1)
        for (i <- 0 until shape(0)) {
          System.arraycopy(data, i * shape(1), resultData, i * resultShape(1), shape(1))
          resultData(i * resultShape(1) + resultShape(1) - 1) = value
        }
        NDArray(resultData, resultShape)
      } else {
        ???
      }
    }

  }

  object NDArray {

    def apply[T: ClassTag](data: Array[T], shape: Array[Int]): NDArray[T] = {
      new NDArray[T](data, shape)
    }

    def apply[T: ClassTag](array: Array[_]): NDArray[T] = {
      def fill(data: Array[T], index: Int, items: Array[_], itemIndex: Int): Int = {
        val item = items(itemIndex)
        item match {
          case values: Array[_] =>
            var dataIndex = index
            var i = 0
            while (i < values.length) {
              dataIndex = fill(data, dataIndex, values, i)
              i = i + 1
            }
            dataIndex
          case _ =>
            data(index) = item.asInstanceOf[T]
            index + 1
        }
      }
      val shape = array.shape()
      val data = Array.ofDim[T](shape.product)
      fill(data, 0, Array(array), 0)
      NDArray(data, shape)
    }

  }

}
