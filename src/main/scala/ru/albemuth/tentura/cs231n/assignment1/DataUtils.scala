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
    val xtr = load(new File(root, "X_train.json"), new TypeReference[Array[Array[Array[Array[Float]]]]]() {})
    val ytr = load(new File(root, "y_train.json"), new TypeReference[Array[Int]]() {})
    val xte = load(new File(root, "X_test.json"), new TypeReference[Array[Array[Array[Array[Float]]]]]() {})
    val yte = load(new File(root, "y_test.json"), new TypeReference[Array[Int]]() {})

    (xtr, ytr, xte, yte)
  }

  def load[T: ClassTag](file: File, typeReference: TypeReference[T]): T = {
    TryWith(new FileInputStream(file)) (in => {
      val mapper = new ObjectMapper
      mapper.readValue(in, typeReference).asInstanceOf[T]
    })
  }

}
