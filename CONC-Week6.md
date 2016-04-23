# Week 6: Concurrency Building Blocks

## Executors
Thread creation is expensive. Therefore, one often *multiplexes threads to perform different task*. The JVM offers abstractions for that `ThreadPool`s and `Executor`s.

 - Threads become in essence the workhorses that perform various tasks presented to them.
 - The number of available threads in a pool is typically some polynomial of the number of cores *N* (e.g. *N^2*).
 - That number is *independent of the number of concurrent activities to perform*.

## Runnables
A task presented to an executor is encapsulated in a `Runnable` object:

```scala
trait Runnable {
  def run(): Unit = {// actions to be performed by the task }
}
```
Runnables can be passed to the `ForkJoinPool` executor. This is a **system provided** thread-pool which also handles tasks spawned by Scala's implementation of parallel collections and Java 8's implementation of streams.
```scala
import java.util.concurrent.ForkJoinPool
object ExecutorsCreate extends App {
  val executor = new ForkJoinPool
  executor.execute(new Runnable {
    def run() = log("This task is run asynchronously")
    })
    Thread.sleep(500)
}
```

 1. Task are run by passing `Runnable` objects to an executor.
 2. There is **no way to await the end of a task** (no .join()). Instead we pause the main thread to give the executor threads time to finish.

In alternative `executor.shutdown()` and `executor.awaitTermination(60, TimeUnit.SECONDS)` (defined by Java's interface `ExecutorService` implemented by `ForkJoinPool`) could be used.

## Execution Contexts
The `scala.concurrent` package defines the `ExecutionContext` trait and object which is similar to `Executor` but more specific to scala.
Execution contexts are passed as implicit parameters to many of Scala's concurrency abstractions. Using the default:

```scala
import scala.concurrent
object ExecutionContextCreate extends App {
  val ectx = ExecutionContext.global
  ectx.execute( new Runnable ).... // as in previous example.
}

// UTILITY METHOD TO CUT THE BOILERPLATE
def execute(body: => Unit) = scala.concurrent.ExecutionContext.global.execute(
  new Runnable { def run () = body})

// use:
execute { log("This task is run asynchronously")
  Thread.sleep(500)}
```

## Atomic Primitives
We now look at the primitives used to realize the higher level operations `wait()`, `notify` and `notifyAll`.

On the JVM they are based on the notion of an ***atomic variable***.

 - An ** *atomic variable* is a memory location that supports *linearizable* operations**
 - A ***linearizable*** operation is one that **appears instantaneously with the rest of the system**.

We also say that the operation is performed **atomically**.
In Java, classes that create atomic variables are defined in the package `java.util.concurrent.atomic` and include `AtomicInteger, AtomicLong, AtomicBoolean, AtomicReference`.

### Using Atomic Variables
Atomic classes have atomic methods such as `incrementAndGet, getAndSet`: they are complex and linearizable operations during which **no intermediate result can be observed by other threads**.

### Compare and Swap
Atomic operations are usually based on the **compare-and-swap (CAS)** primitive: this operation is available as `compareAndSet` on atomic variables. It is usually *implemented by the underlying hardware as a machine instruction*.

We can imagine they are implemented like this:
```scala
def compareAndSet(expect: T, update: T): Boolean = this.synchronized {
  if (this.get == expect) { this.set(update); true}
  else false
}
```

### Programming Without Locks
`synchronized` locks are convenient but they bring about the possibility of deadlocks and they can be arbitrarily delayed by the OS or other threads.

Using atomic variables and their **lock-free operations** we can avoid `synchronized` and its problems: a thread executing a lock-free operation cannot be pre-empted by the OS so **it cannot temporarily block other threads**.

### Simulating Locks
We can implement `synchronized` only from atomic operations:

```scala
private val lock = new AtomicBoolean(false)
def mySynchronized(body: => Unit): Unit = {
  while (!lock.compareAndSet(false, true)) {}
  try body
  finally lock.set(false)
}
```

## Lock-Free operations
Definition of lock-freedom without atomic variables or other primitives:

*An operation op is lock-free if whenever there is a set of threads executing op at least one thread completes the operation after a finite number of steps, regardless of the speed in which the different threads progress.*

Lock-freedom is difficult to reason about!!

## Lazy Values
```scala
@volatile private var x_defined = false
private var x_cached: T = _
def x: T = {
  if (!x_defined) this.synchronized {
    if (!x_defined) { x_cached = E ; x_cached = true}
  }
  x_cached
}
```
This pattern is called *double locking* (double checking would more appropriate)
This is the actual implementation of lazy vals in scalac. However:

 - `synchronized` -> not lock-free
 - if this is already used as a monitor there is the risk of a deadlock
```scala
object A { lazy val x = B.y }
object B { lazy val y = A.x }
```
Sequential execution -> Infinite loop & stack overflow
Concurrent execution -> A waits for B, B waits for A.... Either stack overflow or deadly (not only it does not do what expected, it is **nodeterministc**).

An alternative (used in `dotty`, new scala compiler)
```scala
private var x_evaluating = false
private var x_defined = false
private var x_cached = _

def x: T = {
  if (!x_defined) {
    this.synchronized {
      if (x_evaluating) wait() else x_evaluating = true
    }
    if (!x_defined) {
      x_cached = E
      this.synchronized {
        x_evaluating = false
        x_defined = true
        notifyAll()
      }
    }
  }
  x_cached
}
```
The synchronized blocks are very short, they will not cause deadlocks and they do not use `this` as a monitor inside. Having 2 `synchronized` does not worsen performances (two small instead of one big). Other variants could use different locks but would require allocating other object(s), which would too expensive.

Some perks:
 - The evaluation of E happens outside a monitor (synchronized) -> no arbitrary slowdowns
 - No interference with user defined locks
 - **Deadlocks are still possible, but only in cases where sequential execution would give an infinite loop**

## Using Collections Concurrently
Operations on **mutable** collections are usually not thread safe.

One solution to deal with this is to use `synchronized`
```scala
object CollectionBad extends App {
  val buffer = mutable.ArrayBuffer[Int]()
  def add(numbers: Seq[Int]) = execute {
    buffer ++= numbers
    log("buffer = "+ buffer.toString)
  }
}
// add with synchronized becomes
def add(numbers: Seq[Int]) = execute { buffer.synchronized {
  buffer ++= numbers ; log( ... )}}
```
However `synchronized` often leads to too much blocking because of *coarse-grained locking*.

### Concurrent Collections
To gain speed we can use or implement *concurrent collection*.

#### Example: concurrent queue
Make `head` and `last` atomic. We use an atomic CAS for (`last`) and fix the assignment of `prev` when CAS is successful. Fix `prev.next == null` instead of `prev.next == last1` in `remove`:
```scala
object ConcQueue {
  private class Node[T](@volatile var next: Node[T]) {
    var elem: T = _
  }
}

class ConcQueue[T] {
  import ConcQueue._
  private var last = new AtomicReference(new Node[T](null))
  private var head = new AtomicReference(last.get)

  @tailrec final def append(elem: T): Unit = {
    val last1 = new Node[T](null)
    last1.elem = elem //create new node
    val prev = last.get
    if (last.compareAndSet(prev, last1)) prev.next = last1 // if one modification succeds, do the other
    else append(elem) // start over if fail <-> someone modified prev during lines
  }

  @tailrec final def remove(): Option[T] =
    if (head eq last) None
    else {
      val hd = head.get
      val first = hd.next
      if (first != null && head.compareAndSet(hd, first)) Some(first.elem)
      else remove()
    }
}
```
#### In the library
In `java.util.concurrent` there are some multiple implementations of the interface `BlockingQueue` (blocking => `wait()` such as Assignment 5).

The Scala library also provides implementations of concurrent sets and maps.
