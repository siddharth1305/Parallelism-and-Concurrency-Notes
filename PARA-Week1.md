# Week 1: Parallel Programming

## Intro
*Parallel computing*: many calculations performed at the same time.
Idea: divide computation into smaller subproblems each of which can be solved simultaneously using parallel hardware.

Interest: processor frequency scaling hit the *power wall*, so processor vendors decided to provide multiple CPU cores on the same chip, each capable of executing separate instructions streams.

Parallelism and concurrency are close concepts:

 - **Parallel** program: uses parallel hardware to execute computation more quickly: EFFICIENCY
 - **Concurrent** program: *may or may not* execute multiple executions at the same time. MODULARITY, RESPONSIVENESS, MAINTAINABILITY.

We focus on **task-level parallelism**: executing separate instruction streams in parallel. The hardware we will target is **multi-core processors** and **symmetric multiprocessors**.

## Parallelism on the JVM

### Operating system, processes and multitasking
 - *Operating system* : software that manages hardware and software resources, and schedules program execution.
 - *Process* : an instance of a program that is executing in the OS.

The operating system multiplexes many different processes and a limited number of CPUs, so that each process gets a *time slice* of execution: this is **multitasking**. Two different processes cannot access each other's memory directly, they are **isolated**.

### Threads
Each process can contain multiple *independent concurrenccy units* called **threads**.
They can be started from the same program and **share the same memory address space**. **Each thread has a program counter and a program stack**.
In the JVM threads cannot modify each other's stack memory, only the heap memory.[more here](http://www.journaldev.com/4098/java-heap-memory-vs-stack-memory-difference)

Each JVM process starts with a **main thread**, to add additional threads

 - Define a `Thread` subclass containing a `run` method
 - Instantiate a new object
 - Call `start`
 - When done call `join()`

#### Atomicity
Statements in different threads can overlap. Sometimes we want to ensure that a sequence of statements in specific thread executes at once.

An operation is ***atomic*** if it appears as if it occurred instantaneously from the point of view of other threads.

To achieve atomicity we use the `synchronized` block: code executed in the block is never executed by two threads at the same time.

Example:

```scala
private val x = new AnyRef{}
private var uidCount = 0L
def getUniqueId(): Long = x.syncronized {
   uidCount += 1
   uidCount
}
```

Different threads use the block to *agree on unique values*. It is an example of a *syncronization primitive*.
Invocations of the `syncronized` block can nest.

#### Deadlocks
Deadlocks can be caused by nested `synchronized` blocks
A *Deadlock* occurs when multiple threads compete for resources and wait for each other to finish without releasing the already acquired resources.

One solution is to always acquire resources in the same order.

#### Memory Model
Memory model is a set of rules that describes how threads interact when accessing shared memory.

In the JVM we have the *Java Memory Model*:

 1. Two threads writing to two separate locations in memory do not need synchronization
 2. A thread X that calls `join` on another thread Y is guaranteed to observe all the writes by thread Y after `join` returns

## Running Computations in Parallel
Given two expressions `e1` and `e2` compute them in parallel and return the pair of results:

```scala
val (res1, res2) = parallel(e1, e2)
val ((res1, res2), (res3, res4)) = parallel(parallel(e1, e2), parallel(e3, e4))

// The parallel function's signature
def parallel[A, B](taskA: => A, taskB: => B): (A, B) = { ... }
// NOTE: call by name to pass unevaluated computations or we wouldn't have parallel computations
```

### Under the hood
Efficient parallelism require language and libraries, virtual machine, operating system and hardware support.

`parallel` uses JVM threads which *typically* map to OS threads which can be scheduled on different cores.

Given sufficient resources, a parallel program can run faster.
Note that thanks to the different layers of abstractions a program written for parallel hardware can run even when there is only one processor core available (without speedup).

**Hardware is important**: **memory (RAM) is bottleneck** that can void all the effort necessary to write parallel software.

The running time of `parallel(e1, e2)` is the maximum of the two running times.

## First-Class Tasks
As we have just seen, the running time of `parallel(e1, e2)` is the maximum of the two running times. When need a more flexible construct, that does not wait for the end of both threads to return.
We use the `task` construct:

```scala
val t1 = task(e1)
val t2 = task(e2)
val v1 = t1.join //blocks and waits until the result is computed
val v2 = t2.join //obtain the result of e1
```

`task` start computation in *the background*. Subsequent calls to `join`return the same result

```scala
def task(c: => A): Task[A]

trait Task[A] {
   def join: A
}
```
We could omit `.join` by defining an implicit conversion: `implicit def getJoin[T](x: Task[T]): T = x.join`, then `task` will return the result.

`task` can be used to define `parallel`:

```scala
def parallel[A, B](cA: => A, cB: => B): (A, B) = {
   val tB: Task[B] = task { cB  } // if we called .join here the two threads would be executed sequentially and not in parallel
   val tA: A = cA
   (tA, tB.join)
}

```

## How Fast are Parallel Programs?

## Benchmarking Parallel Programs



[Next week lecture notes](PARA-Week2.md)
