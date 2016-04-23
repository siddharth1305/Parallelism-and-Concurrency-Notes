# Week 5: Introduction to Concurrent Programming

In concurrent programming a program is:

 - A set of **concurrent** computations
 - That execute during **overlapping time intervals**
 - That **need to coordinate** in some way.

Concurrency is hard: RISKS = non determinism + race conditions + deadlocks

## Advantages:

 1. Performances (thanks to parallelism in hardware)
 2. Responsiveness (faster I/O **without blocking** or polling)
 3. Distribution (programs can be spread over multiple nodes)

## Terminology
The operating systems schedules *cores* to run *threads*. Threads in the same process *share memory*.

Week1 -> how to start threads in the JVM

In scala a helper method is provided to create threads:

```scala
def thread(b: => Unit) = {
  val t = new Thread {
    override def run() = b
  }
  t.start()
  t
}

val t = thread { println("New thread running")}
t.join()
```

## Threads on the JVM
A thread image in memory contains *copies of the processor registers* and the *call stack (~2MB)*.

The os has a *scheduler* that runs all active threads on all cores. If there are more threads than cores, they are *time sliced*.

Switching from a thread to another (***context switching***) is a complex and expensive operation (~1000ns = 1  per switch) but is less expensive that blocking io.

Threads can be paused **from inside** with `Thread.sleep(time)`.

## Interleaving and locking
Threads operations are executed concurrently or interleaved: there could be a context switch (ordered by the processor scheduler for example) in the middle of the execution of a thread. This could cause non-determinism.

Atomicity and synchronize were covered in week1.

```scala
def synchronized[T](block: => T): T

obj.syncronized { block }
```

Here `obj` acts as a ***lock***: it makes sure that no other thread executes inside a syncronized method ***of the same object***.

 - `block` can be executed only by the thread holding the lock
 - **only one thread can hold a lock at any one time**.

Every object can serve as a lock, which is expensive when not on the JVM (e.g. Scala.js) therefore in the future `synchronized` will become a method of the trait `Monitor` whose instances only will be able to serve as locks.

## A Memory Model

### Sequential Consistency Model:
The result of any execution is the same as if the operations of all the processors were executed in some sequential order, and the operations of each individual processor appear in this sequence in the order specified by its program.

### Forget it
However **modern multicore processors do not implement the sequential consistency model**. The problem is caused by **caches**:

 - Each core might have a different copy of shared memory in its private caches
 - Write-backs of caches to main-memory happens at *unpredictable* times.

Also compilers: they are often allowed to reorder instructions.

As a general rule: ***Never write to a variable being read by another thread && Never try to read a variable being written by another thread***.

`synchronized` ensures atomic execution and that **writes are visible**:
 - after `obj.synchronized { ... }` was executed by thread `t`, all writes of `t` up to that point are visible to every thread that subsequently executes a `obj.synchronized {... }`

## Thread coordination
Monitors can do more than just locking with `syncronized`, they offer the methods:

 - `wait()`: suspends the current thread. it *releases the lock* so other threads can enter the monitor.
 - `notify()` wakes up **one thread** waiting on the current object
 - `notifyAll()` wakes up **all threads** waiting on the current object.

They **should only be called from within a synchronized on `this`**.
`notify()` and `notifyAll` schedule other threads for execution after the calling thread has released the lock (left the monitor).

See assignment 5 for an example.

### Signals
The original model of *Monitors* by Per Brinch Hansen has synchronized but not `wait, notify, notifyAll`.
Instead it uses ***signals***:

```scala
class Signal {
  def wait(): Unit // wait for someone
  def send(): Unit // wakes up first waiting thread
}

// Example
class OnePlaceBuffer[Elem] extends Monitor {
  var elem: Elem = _
  var full = false
  val isEmpty, isFull = new Signal
  def put(e: Elem): Unit = synchronized {
    while (full) isEmpty.wait()
    isFull.send()
    full = true; elem = e
  }
  def get(): Elem = synchronized {
    while (!full) wait()
    isEmpty.send()
    full = false; elem
  }
}
```

### Stopping Thread
A method `stop()` exists in `Thread` but is **deprecated and should not be used**:

The thread will be killed **at any arbitrary time (when the message is received)**. It may happen during a lock, leaving the system in a completely undetermined state.

## Volatile Variables
Sometimes we only need ***safe publication*** (instead of atomic publication): in these case there is a **cheaper solution** than using a synchronized: ***volatile fields***:

```scala
@volatile var found = _
val t1 = thread {... ; found = true}
val t2 = thread {while(!found) ...}
```

 - Assignments to the volatile variable **will not be reordered** with respect to other statements in the thread
 - Assignments to the volatile variable are **visible immediately to all other threads**.

## Memory Models in general
A memory model is an abstraction of the hardware capabilities of different computer systems.
It abstracts over the underlying system's *cache coherence protocol*. Memory models are **non-deterministic** to allow some freedom of implementation in hardware or compiler.
Every model is a compromise:

 - More guarantees => easier to write concurrent programs
 - Fewer guarantees => more possibilities for optimization.

### The Java Memory Models
 - **Program order**: each action in a thread happens before every subsequent action in the same thread.
 - **Monitor locking**: unlocking a monitor happens before every subsequent locking of that monitor
 - **Volatile fields**: a write to a volatile field happens before every subsequent read of that field
 - **Thread start**: a call to `start()` on a thread happens before all actions of that thread.
 - **Thread termination**: an action in a thread happens before another thread completes a `join` on that thread
 - **Transivity**: If A happens before B and B happens before C, then A happens before C
