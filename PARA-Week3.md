[Previous week lecture notes](PARA-Week2.md)


# Week 3: Data-Parallelism
## Data-Parallel Programming
We now turn our attetion to **data-parallel** programming: *a form of parallelization that distributes data across computing nodes*.

The simplest form of data-parallel programming is the parallel `for` loop:

```scala
for (i <- (0 until array.length).par) do stuff
```

This loop **is not** functional: it only affects the program through side effects but **as long as iterations write to *separate* memory location, the program is correct**.

### Workload
Different data-parallel programs have different workloads: *Workload* is a function that maps each input element to the amount of work required to process it.
It may be uniform (therefore easy to parallelize) or irregular **depending on the problem instance**.

A *data-parallel scheduler* is therefore needed to efficiently balance the workload across computing units without any knowledge of the workload of each element.

## Data-Parallel Operations I
In scala the `.par` converts a sequential collection to a parallel collection.

However **some operations are not parallelizable**.

### Non-Parallelizable Operations
The function

```scala
def sum(xs: Array[Int]): Int = xs.par.foldLeft(0)(_ + _)
```

cannot be executed in parallel.

**SEQUENTIAL** :`foldRight, foldLeft, reduceRight, scanLeft, scanRight` must process the elements sequentially

**PARALLEL** : `fold` can process elements in a reduction tree and can therefore execute in parallel.

### Some use-cases for `fold`

```scala
def sum(xs: Array[Int]): Int = xs.par.fold(0)(_ + _)

def max(xs: Array[Int]): Int = xs.par.fold(Int.MinValue)(math.max)
```

### Prerequisite for `fold`
For `fold` to give us the expected result, we need to provide it with an **associative function**. Commutativity is not sufficient.

In particular the following conditions on `f` must be satisfied:

 1. `f(a, f(b, c) == f(f(a, b), c)`
 2. `f(z, a) == f(a, z) == a` (`z` being the neutral element)

We say that **the neutral element `z` and the binary operation `f` form a *monoid***.

We remark that **commutativity (`f(a,b) == f(b,a)`) does not matter for `fold`**

### Limitations of `fold`
`fold` can only produce values of the same type as the collection that it is called on! Counting elements of a collection is impossible.

We need a more general (*expressive*) operations

### The `aggregate` operation

```scala
def aggregate[B](z: B)(f: (B, A) => B, g: (B, B) => B): B
```

`f` produces the intermediary collections which will be combined by `g` to produce the final result. To count vowels now:

```scala
text.par.aggregate(0)( (count, chara) => if (isVowel(c)) count + 1 else count, _ + _ )
```

So far we studied *accessor combinators*

Operations like `map, filter, flatMap, groupBy` do not return a single value but instead return new collections as results. They are called *Transformer combinators*. `f`.

## Scala Parallel Collections
Here are the main types of scala collections and their parallel counterparts:

|               Description               |    Sequential    |     Parallel     |     Agnostic     |
|:---------------------------------------:|:----------------:|:----------------:|:----------------:|
| operations implemeted using `foreach`   | `Traversable[T]` |                  |                  |
| operations implemented using `iterator` | `Iterable[T]`    | `ParIterable[T]` | `GenIterable[T]` |
| ordered sequence                        | `Seq[T]`         | `ParSeq[T]`      | `GenSeq[T]`      |
| unordered collection, no duplicates     | `Set[T]`         | `ParSet[T]`      | `GenSet[T]`      |
| keys -> values, no duplicate keys       | `Map[K ,V]`      | `ParMap[K ,V]`   | `GenMap[K ,V]`   |

`Gen*` allows us to write code that is **agnostic** about parallelism: code that is *unaware* of parallelism.  `.par` converts a sequential collection into a parallel one.

| Sequential Collection  | Corresponding Parallel Collection |
|------------------------|-----------------------------------|
| `ParArray[T]`          | `Array[T]`and `ArrayBuffer[T]`    |
| `Range`                | `ParRange`                        |
| `Vector[T]`            | `ParVector[T]`                    |
| `immutable.HashSet[T]` | `immutable.ParHashSet[T]`         |
| `immutable.HashMap`    | `immutable.ParHashMap[K, V]`      |
| `mutable.HashSet[T]`   | `mutable.ParHashSet[T]`           |
| `mutable.HashMap[T]`   | `mutable.ParHashMap[K, V]`        |
| `TrieMap[K, V]`        | `ParTrieMap[K, V]`                |

Another interesting parallel data structures is `ParTrieMap[K, V]`: a **thread-safe** parallel map with **atomic snapshots** (counterparto of `TrieMap`).

For other collections, `.par` creates the closest parallel collection: `List` is converted to `ParVector`.

### Side effecting operations

```scala
def intersection(a: GenSet[Int], b: GenSet[Int]): Set[Int] = {
  val result = mutable.Set[Int]()
  for (x <- a) if (b contains x) result += x
  result
}
```

This code is not safe for parallelism! **Avoid mutations to the same memory locations without proper synchronization**. However synchronization might not be the solution as it has its own side effects.

Side effects can be avoided by using the correct *combinators* (`filter....`):

```scala
def intersection(a: GenSet[Int], b: GenSet[Int]): Set[Int] = {
  if (a.size < b.size) a.filter(b(_))
  else b.filter(a(_))
}
```

### Concurrent Modifications during Traversals
Rule: **Never modify a parallel collection on which a data-parallel operation is in progress**

 - Never write to a collection that is concurrently traversed
 - Never read from a collection that is concurrently modified

In either case, propgram **non-deterministically** prints different results or crashes.

### The `TrieMap` collection

`TrieMap` is an exception to these rules as its method `snapshot` can be used to efficiently grab the current state:

```scala
val graph = concurrent.TrieMap[Int, Int]() ++= (0 until 100000).map(i => (i, i + 1))
graph(graph.size - 1) = 0
val previous = graph.snapshot()
for ((k, v) <- graph.par) graph(k) = previous(k)
```

This code works as expected.


## Splitters and Combiners
Let's now focus on the following abstractions:

 - Iterators
 - Splitters
 - Builders
 - Combiners

### Iterator
The simplified trait is

```scala
trait Iterator[A] {
   def next(): A
   def hasNext(): Boolean
}

def iterator: Iterator[A] // on every collection
```

The *iterator contract*:
 - `next` can be called only if `hasNext` returns `true`
 - after `hasNext` returns `false`, it will always return `false`

use:

```scala
def foldLeft[B](z: B)(f: (B, A) => B): B = {
  var s = z
  while (hasNext) s = f(s, next())
}
```

### Splitter
The `Splitter` trait:

```scala
trait Splitter[A] extends Iterator[A] {
  def split: Seq[Splitter[A]]
  def remaining: Int
}

def splitter: Splitter[A] // on every parallel collection
```

the contract:

 - After calling `split` the original splitter is left in an *undefined state*
 - the resulting splitters traverse *disjoint subsets* of the original splitter
 - `remaining` is an *estimate* on the number of remaining elements
 - `split` is *efficient* : O(log *n*) or better

use:

```scala
def fold(z: A)(f: (A, A) => A): A = {
  if (remaining < threshold) foldLeft(z)(f)
  else {
    val children = for (child <- split) yield task {child.fold(z)(f)}
    children.map(_.join()).foldLeft(z)(f)
  }
}
```

### Builder
trait:

```scala
trait Builder[A, Repr] {
  def +=(elem: A): Builder[A, Repr]
  def result: Repr
}

def newBuilder: Builder[A, Repr] // on every collection
```

Contract:

 - calling `result` returns a collection of type `Repr`, containing the elements that were previously added with `+=`
 - calling `result` leaves the `Builder` in an *undefined state*

use:

```scala
def filter(p: T=> Boolean): Repr = {
  val b = newBuilder
  for (x <- this) if (p(x)) b += x
  b.result
}
```

### Combiner
trait:

```scala
trait Combiner[A, Repr] extends Builder[A, Repr] {
  def combine(that: Combiner[A, Repr]): Combiner[A, Repr]
}

def newCombiner: Combiner[T, Repr] //on every parallel collection
```

contract:

 - calling `combine` returns a *new combiner* that contains elements of the input combiners
 - calling `combine` leaves both original `Combiner`s in an *undefined state*
 - `combine` is efficient: O(log *n*) or better

use:

Example during class

[Next week lecture notes](PARA-Week4.md)
