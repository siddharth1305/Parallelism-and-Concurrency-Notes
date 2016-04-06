[Previous week lecture notes](PARA-Week1.md)


# Week 2: Basic Task Parallel Algorithms

## Parallel Merge-Sort:
Use divide and conquer:

 1. Recursively sort the two halves of the array in parallel

```scala
def sort(from: Int, until: Int, depth: Int): Unit = {
  if (depth == maxDepth) {
     quickSort(xs, from, until - from)
  } else {
     val mid = (from + until) / 2
     parallel(sort(mid, until, depth + 1), sort(from, mid, depth + 1))
     val flip = (maxDepth - depth) % 2 == 0
     val src = if (flip) ys else xs
     val dst = if (flip) xs else ys
     merge(src, dst, from, mid, until)
  }
}
sort(0, xs.length, 0)
```

 2. Sequentially merge the two halves by using a temporary array
 3. Copy the temporary array into the original one

```scala
def copy(src: Array[Int], target: Array[Int],
  from: Int, until: Int, depth: Int): Unit = {
	if (depth == maxDepth) {
		Array.copy(src, from, target, from, until - from)
	} else {
		val mid = (from + until) / 2
		val right = parallel(
		  copy(src, target, mid, until, depth + 1),
		  copy(src, target, from, mid, depth + 1))
	}
}
if (maxDepth % 2 == 0) copy(ys, xs, 0, xs.length, 0)
```

## Data Operations and Parallel Mapping
We need to process collections in parallel but this can be done only if some properties of collections and of the operation we want to apply are satisfied.

### Functional programming and collections
Operations such as `map`, `fold` and `scan` are key to functional programming but become even more important for parallel collections: they allow write simpler parallel programs.

As always, the choice of the data structure (collection) to use is key: **Lists** are not good for parallel implementations:

 - Splitting is slow (linear search for the middle)
 - Concatenation is slow (linear search for the end)

Therefore we use:

 - **Arrays**, which are imperative
 - **Trees**, which can be implemented functionally

### Parallelizing Map on arrays
We need to parallelize both the computations of the function and the access to the elements to apply the function.

```scala
def mapASegPar[A,B](inp: Array[A], left: Int, right: Int, f : A => B,
                    out: Array[B]): Unit = {
	// Writes to out(i) for left <= i <= right-1
	if (right - left < threshold)
		mapASegSeq(inp, left, right, f, out)
	else {
		val mid = left + (right - left)/2
		parallel(mapASegPar(inp, left, mid, f, out),
	                 mapASegPar(inp, mid, right, f, out))
	}
```

Remarks:

- **writes need to be disjoint**, or unpredictable output (nondeterministic behavior)
- **thresholds need to be large enough** or we lose efficiency

### Parallelizing Map on immutable trees

Assume:

```scala
sealed abstract class Tree[A] { val size: Int }
// Leaves store array segments
case class Leaf[A](a: Array[A]) extends Tree[A] {
	override val size = a.size
}
// Non-leaf node stores two subtrees
case class Node[A](l: Tree[A], r: Tree[A]) extends Tree[A] {
	override val size = l.size + r.size
}
```

If trees are balanced, we can explore branches in parallel efficiently:

```scala
def mapTreePar[A:Manifest,B:Manifest](t: Tree[A], f: A => B) : Tree[B] =
t match {
	case Leaf(a) => {
		val len = a.length; val b = new Array[B](len)
		var i= 0
		while (i < len) { b(i)= f(a(i)); i= i + 1 };
		Leaf(b)
		}
	case Node(l,r) => {
		val (lb,rb) = parallel(mapTreePar(l,f), mapTreePar(r,f))
		Node(lb, rb)
		}
}
```
Which will run in O(height of tree)

### Arrays or immutable trees?

| **Arrays**                                                         | **Immutable Trees**                                                                  |
|--------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| - Imperative: must ensure parallel task write to disjoint paths    | + Purely functional: no need to worry about disjointness of writes by parallel tasks |
| + Good memory locality                                             | - Bad locality                                                                       |
| - Expensive to concatenate                                         | + efficient combination of two trees                                                 |
| + random access to elements, on shared memory can share same array | - High memory allocation overhead                                                    |

## Fold (Reduce) Operations
Seen `map`, we focus on `fold`: combining elements with a given operation

`fold` takes a binary operations, but variants differ whether they take an initial element or assume non-empty list, in which order to combine operations of collection.

To parallelize we focus on **associative operations** such as addition and string concatenation (not subtraction)

### Associative operations
`f: (A,A) => A` is associative **iff** for every *x,y,z* *f(x, f(y,z)) = f(f(x,y),z)*

Then if we have two expressions with same list of operands but different parentheses, these expression **evaluate to the same result**.

We can represent expression with trees: each leaf is a value, each node a computation of the operation.

#### Folding (reducing) trees in parallel

```scala
sealed abstract class Tree[A]
case class Leaf[A](value: A) extends Tree[A]
case class Node[A](left: Tree[A], right: Tree[A]) extends Tree[A]

def reduce[A](t: Tree[A], f : (A,A) => A): A = t match {
	case Leaf(v) => v
	case Node(l, r) => {
		val (lV, rV) = parallel(reduce[A](l, f), reduce[A](r, f)) // remove parallel word to have the sequential version
		f(lV, rV)
	}
}
```

The complexity is again the height of the tree.
If the operation is not associative, the result depends on the structure of the tree.

### Folding (reducing) arrays
To reduce an array we can convert it into a balanced tree and then do tree reduction. If the algorithm is "simple enough" we do not need a formal definition of the tree, just divide it into halves:

```scala
def reduceSeg[A](inp: Array[A], left: Int, right: Int, f: (A,A) => A): A = {
	if (right - left < threshold) {
		var res= inp(left); var i= left+1
		while (i < right) { res= f(res, inp(i)); i= i+1 }
		res
	} else {
		val mid = left + (right - left)/2
		val (a1,a2) = parallel(reduceSeg(inp, left, mid, f),
		reduceSeg(inp, mid, right, f))
		f(a1,a2)
	}
}
def reduce[A](inp: Array[A], f: (A,A) => A): A =
reduceSeg(inp, 0, inp.length, f)
```

`map` can be combined with reduce to avoid intermediate collections

## Associative Operation
Reminder: `f: (A,A) => A` is associative **iff** for every *x,y,z* *f(x, f(y,z)) = f(f(x,y),z)*

Consequence (the two are equivalent):

 - two expressions with the same list of operands connected with the operation but different parentheses evaluate to the same result
 - reduce on any tree with this list of operands gives the same result

### False friends: commutativity
`f: (A,A) => A` is commutative iff for every *x,y* *f(x,y) = f(y,x)*

 - **Associativity does not imply commutativity**
 - **Commutativity does not imply associativity**

**for correctness of `reduce` associativity is sufficient**


#### Examples of operations that are bot associative and commutative

 - addition and multiplication
 - addition and multiplication modulo a positive integer
 - union, intersection, symmetric difference
 - union of multisets preserving duplicate elements
 - boolean || && xor
 - addition and multiplication of polynoials
 - addition of vectors
 - addition of matrices of fixed dimension

#### Examples of operations that are associative but not commutative

 - list concatenation
 - string concatenation
 - matrix multiplication of compatible dimensions
 - composition of relations
 - composition of functions

**Many operations are commutative but not associative** : *f(x,y) = x^2 + y^2*

### Mapping does not preserve associativity
if *f* is commutative and associative and *g,h* are arbitrary functions then

*i(x,y) = h(f(g(x), h(y))) = h(f(g(y), h(x))) = i(y,x)*

is commutative but if often loses *f*'s associativity.

**floating point addition is commutative but not associative**

### Making an operation commutative is easy!
Suppose binary operation `g` and a strict total ordering `less`. Then

```scala
def f(x: A, y: A) = if (less(y,x)) g(y,x) else g(x,y)
```

is commutative. **There is no such trick for associativity**.

### Example: average

```scala
f((sum1, len1), (sum2, len2)) = (sum1 + sum2, len1 + len2)

val sum (sum,length) = reduce(map(collection, (x: Int) => (x,1)), f)
sum/length
```

### Associativity thanks to symmetry and commutativity

 If

 - Commutativity
 - *f(f(x,y), z) = f(f(y,z), x)*

are satisfied then *f* is also **associative**.

## Parallel Scan Left
Now we turn our attention to **scanLeft** which produces a list of the folds of all list prefixes: `List(1,3,8).scanLeft(100)((s,x) => s + x) == List(100, 101, 104, 112)`
We assume that the operation is **associative**.

### Sequential definition

```scala

def scanLeft[A](inp: Array[A],
                a0: A, f: (A,A) => A,
		out: Array[A]): Unit = {
	out(0) = a0
	var a = a0
	var i = 0
	while (i < inp.length) {
		a= f(a,inp(i))
		i= i + 1
		out(i)= a
	}
}
```

### Towards parallelization using trees
We have to give up on reusing all intermediate results:

 - do more `f` applications
 - improve parallelism, to compensate the additional `f` applications

To reuse some of the intermediate results, we remember that `reduce` proceeds by applying the operations in a tree, so let's assume that the input is also a tree.

```scala
// input tree class
sealed abstract class Tree[A]
case class Leaf[A](a: A) extends Tree[A]
case class Node[A](l: Tree[A], r: Tree[A]) extends Tree[A]

//result tree class
sealed abstract class TreeRes[A] { val res: A }
case class LeafRes[A](override val res: A) extends TreeRes[A]
case class NodeRes[A](l: TreeRes[A], override val res: A, r: TreeRes[A]) extends TreeRes[A]
```

We now need to transform the input tree into the output tree:

```scala
def upsweep[A](t: Tree[A], f: (A,A) => A): TreeRes[A] = t match {
	case Leaf(v) => LeafRes(v)
	case Node(l, r) => {
		val (tL, tR) = parallel(upsweep(l, f), upsweep(r, f))
		NodeRes(tL, f(tL.res, tR.res), tR)
	}
}

```

And to use this intermediate tree to produce a `Tree` whose leaves are the result

```scala
def downsweep[A](t: TreeRes[A], a0: A, f : (A,A) => A): Tree[A] = t match {
case LeafRes(a) => Leaf(f(a0, a))
	case NodeRes(l, _, r) => {
		val (tL, tR) = parallel(downsweep[A](l, a0, f),
		downsweep[A](r, f(a0, l.res), f))
		Node(tL, tR)
		}
}
```

and `scanLeft` becomes easy to define

```scala
def scanLeft[A](t: Tree[A], a0: A, f: (A,A) => A): Tree[A] = {
	val tRes = upsweep(t, f)
	val scan1 = downsweep(tRes, a0, f)
	prepend(a0, scan1)
}

def prepend[A](x: A, t: Tree[A]): Tree[A] = t match {
	case Leaf(v) => Node(Leaf(x), Leaf(v))
	case Node(l, r) => Node(prepend(x, l), r)
}
```

### Using arrays
To make it more efficient, we use trees that have arrays in leaves instead of individual elements.



[Next week lecture notes](PARA-Week3.md)
