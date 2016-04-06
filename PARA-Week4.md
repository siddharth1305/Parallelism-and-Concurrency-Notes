[Previous week lecture notes](PARA-Week3.md)


# Week 4: Data Structures for Parallel Computing

## Implementing Combiners
 - `conbine` represents union for a set or a map and concatenation for a sequence.

 - `combine` must be **efficient** i.e. *O*(log *n* + log *m*) (*n,m* being the size of the 2 input collections)

### Arrays
cannot be efficiently concatenated.

### Sets
Sets have have efficient lookup, insertion and deletion. The running time depends on the implementation:

 - Hash tables, *expected* O(1)
 - Balanced trees: O(log *n*)
 - Linked lists: O(*n*)

However, **most implementations do not have efficient union operation**.

### Sequences
Complexities depend on the implementation:

 - Mutable linked lists: O(1) append and prepend (and concatenaton), O(*n*) insertion
 - functional (cons) lists: O(1) prepend, everything else (concatenaton) O(*n*)
 - Array Lists: amortized O(1) append, O(1) random acces, everything else (concatenaton) O(*n*)

## Parallel Two-Phase Construction
To avoid these unefficient methods, most data structures can be constructed using ***two-phase construction***.

This technique relies on an *intermediate data structures* that:

 - has an efficient ( O(log *n* + log*m*) or better) `combine`
 - has an efficient `+=`
 - can be converted to the resulting data structures in O(*n/P*)

### Two-phase construction for Arrays
The two phases are implemented in `result` and they are:

 1. partition the indices into subintervals
 2. initialize array in **parallel**

```scala
class ArrayCombiner[T <: AnyRef: ClassTag](val parallelism: Int) {
  private var numElems = 0
  private val buffers = new ArrayBuffer[ArrayBuffer[T]]
  buffers += new ArrayBuffer[T]

  // O(1) amortized, low constat factors ~ arraybuffer
  def +=(x: T) {
    buffers.last += x
    numElems += 1
    this
  }

  //O(P) if buffers contains less than O(P) nested arraybuffers
  def combine(that: ArrayCombiner[T]) = {
    buffers ++= that.buffers
    numElems += that.numElems
    this
  }

  def result: Array[T] = {
    val array = new Array[T](numElems)
    val step = math.max(1, numElems / parallelism)
    val starts = (0 until numElems by step) :+ numElems
    val chunks = starts.zip(starts.tail)
    val tasks = for ((from, end) <- chunks) yield task {
      copyTo(array, from, end)
    }
    task.foreach(_.join())
    array
  }
}
```

### Two-phase Construction for Hash Tables

 1. Partition the hash codes into buckets
 2. allocate the table, map hash codes from different buckets into different regions

### Two-phase Construction for Search Trees

 1. Partition the elements **into non-overlapping intervals** according to their ordering
 2. Construct search trees in **parallel** and link non-overlapping trees.

### Two-phase Construct for Spatial Data Structures

 1. Spatially partition the elements
 2. Construct non **non-overlapping** subsets and link them

## Implementing combiners
Two phase construction is one of the three main strategies:

 1. Two-phase construction: The combiner uses an intermediate data structure with an efficient `combine` method to partition the elements. Then `result` is called to construct the final data structure **in parallel** from the intermediate data structure.
 2. An efficient concatenation or union operation (*algorithm*): the preferred way **when the resulting data structure allows it**
 3. *Concurrent data structure*: different combiners share the same underlying data structure and rely on **synchronization** to correctly update the data structure when `+=` is called.


[Next week lecture notes](CONC-Week1.md)
