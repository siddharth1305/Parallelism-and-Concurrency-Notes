# Week 7: Try and Future
|                  | One         | Many            |
|------------------|-------------|-----------------|
| **Synchronous**  | `T/Try[T]`  | `Iterable[T]`   |
| **Asynchronous** | `Future[T]` | `Observable[T]` |

## Try[T], a monad that handles exceptions
We use this monad to **encapsulate exceptions inside objects**: normally they work only in one thread, encapsulation allows us to move them between threads.

Let's consider the simple adventure game:
```scala
trait Adventure {
  def collectCoins(): List[Coin]
  def buyTreasure(coins: List[Coin]): Treasure
}

val adventure = Adventure()
val coins = adventure.collectCoins()
val treasure = adventure.buyTreasure(coins)
```
We might need to throw an exception in `collectCoins`:
```scala
def collectCoins(): List[Coin] = {
  if (eatenByMonster(this)) throw new GameOverException("oops")
  List(Gold, Gold, Silver)}
```
But then the return type would be wrong! Accept that failure may happen:
```scala
abstract class Try[T]
case class Success[T] (elem: T) extends Try[T]
case class Failure(t: Throwable) extends Try[Nothing]

// Then our adventure:
trait Adventure {
  def collectCoins(): Try[List[Coin]]
  def buyTreasure(coins: List[Coin]): Try[Treasure]
}
```
### Higher-order functions to manipulate Try[T]
 - `def flatMap[S](f: T=>Try[S]): Try[S]`
 - `def flatten[U <: Try[T]]: Try[U]`
 - `def map[S](f: T=>S): Try[T]`
 - `def filter(p: T=>Boolean): Try[T]`
 - `def recoverWith(f: PartialFunction[Throwable, Try[T]]): Try[T]`

They allow us `val treasure: Try[Treasure] = adventure.collectCoins().flatMap(coins => adventure.buyTreasure(coins))`
We can also use for-comprehension:
```scala
val treasure: Try[Treasure] = for {
  coins     <- adventure.collectCoins()
  treasure  <- buyTreasure(coins)
} yield treasure
```
### Under the hood: map
```scala
def map[S](f: T=>S): Try[S] = this match {
  case Success(value)     => Try(f(value))
  case failure@Failure(t) => failure
}

object Try {
  def apply[T](r: => T): Try[T] = {
    try { Success(r) }
    catch { case t => Failure(t) }
  }
}
```
## Future[T], a monad that handles exceptions and latency
In the future, not now, the computation will be done. You will either have a result or an exception.
Consider:
```scala
trait Socket {
  def readFromMemory(): Array[Byte]
  def sendToEurope(packet: Array[Byte]): Array[Byte]
}
```
It looks a lot like the previous game. However here some `readFromMemory` and `sendToEurope` are operations that ***take time*** (50 000 and 150 000 000 ns). It would be a huge waste to have the computantion waiting for these actions to complete before proceeding. We could even wait all that time to discover that the action *failed* resulting in an exception.

Therefore we use `Future[T]`:
```scala
trait Future[T] {
  // we will ignore the implicit execution context
  def onComplete(callback: Try[T] => Unit)(implicit executor: ExecutionContext): Unit
}
```
We can just pass a block of code: `def/va name = Future {action returning T}`

The method `onComplete` is called when the Future is ready. Future is ***asynchronous***.
`callback` needs to use pattern matching to distinguish if the Try is a success or a failure. This however introduces a lot of boilerplate code.

Thus we consider some alternative designs:
```scala
def onComplete(success: T => Unit, failed: Throwable => Unit): Unit
```
Which is what happens with javascript promises: we pass to the function the actions to perform both in case of success and of failure. In javascript these actions are *closures* simple function. In scala they are objects: these are just two sides of the same coin:

 - An *object* is a closure with multiple methods
 - A *closure* is an object with a single method

We can adapt our program:
```scala
trait Socket {
  def readFromMemory(): Future[Array[Byte]]
  def sendToEurope(packet: Array[Byte]): Future[Array[Byte]]
}
```
Using `onComplete` as defined can be cumbersome:
```scala
val confirmation: Future[Array[ByteA]] = packet.onComplete {
  case Success(p) => ... // continue
  case Failure(t) => ...
}
```
**The type do not match**: `onComplete` returns `Unit`. We could
```scala
packer.onComplete {
  case Success(p) => {
    val confirmation = .... // The rest of the application
  }
}
```
all our program would have to move inside this case. UGLY (javascript)

### The solution: flatMap!
```scala
val packet: Future[Array[Byte]] = socket.readFromMemory
val confirmation: Future[Array[Byte]] = packet.flatMap(p => socket.sendToEurope(p))
```

### Robust solution for packet sending
```scala
//This can have Failures inside
def sendTo(url: URL, packet: Array[Byte]): Future[Array[Byte]] =
  Http(url, Request(packet))
    .filter(response => response.isOK)
    .map(response => response.toByteArray)

//use recoverWith to handle exceptions if they are there
def sendToSafe(packet: Array[Byte]): Future[Array[Byte]] =
  sendTo(mailServer.europe, packet) recoverWith {
    case europeError => sendTo(mailServer.usa, packet) recover {
      case usaError => usaError.getMessage.toByteArray
    }
  }
```

### An auxiliary function to reduce boilerplate
```scala
def fallbackTo(that: =>Future[T]): Future[T] = {
  //if this future fails take the successful result of that future
  //if that future fails too, take the error of this future
  this recoverWith {
    case _ => that recoverWith (case _ => this)
  }
}
| |
 V
def sendToSafe(packet: Array[Byte]): Future[Array[Byte]] =
  sendTo(mailServer.europe, packet) fallbackTo {
    sendTo(mailServer.usa, packet)
  } recover {
    case europeError => europeError.getMessage.toByteArray
  }

### Creating futures
```scala
object Future {
  def apply(body: => T)(implicit ....) : Future[T]
}
```

### Future can block, if necessary
```scala
Trait Awaitable[T] extends AnyRef {
  abstract def ready(atMost: Duration): Unit
  abstract def result(atMost: Duration): T
}

trait Future[T] extends Awaitable[T] ....

// Then
val confirmation: Future[Array[Byte]] = packet.flatMap(socket.sendToSafe(_))
val c = Await.result(confirmation, 2 seconds)
```
`Future` can also block our program until the computation is finished. However this is to be avoided in a good reactive program.

#### Because you can, in Scala (almost an anecdote)
The Scala library provides a duration object:
```scala
import scala.language.postfixOps

object Duration {
  def apply(length: Long, unit: TimeUnits): Duration
}
val fiveYears = 1826 minutes
```
