package ru.albemuth.tentura.cs231n

/**
  * @author Vladimir Kornyshev { @literal <gnuzzz@mail.ru>}
  */
object TryWith {
  def apply[A <: AutoCloseable, B](resource: A)(block: A => B): B = {
    var t: Throwable = null
    try {
      block(resource)
    } catch {
      case x => t = x; throw x
    } finally {
      if (resource != null) {
        if (t != null) {
          try {
            resource.close()
          } catch {
            case y => t.addSuppressed(y)
          }
        } else {
          resource.close()
        }
      }
    }
  }
}
