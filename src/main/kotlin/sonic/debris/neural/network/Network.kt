package sonic.debris.neural.network

import koma.*
import koma.extensions.emul
import koma.extensions.get
import koma.extensions.mapIndexed
import koma.matrix.Matrix
import koma.util.validation.validate

fun sigmoid(x: Matrix<Double>): Matrix<Double> =  (exp(-x) + 1.0).epow(-1)
fun sigmoidDerivative(x: Matrix<Double>): Matrix<Double> = sigmoid(x) emul (-sigmoid(x) + 1.0)
fun quadraticCostDerivative(x: Matrix<Double>, y: Matrix<Double>): Matrix<Double> =  x - y

data class Case(val input: Matrix<Double>, val output: Matrix<Double>)
data class Gradients(val ws: List<Matrix<Double>>, val bs: List<Matrix<Double>>)

typealias Function1 = (x: Matrix<Double>) -> Matrix<Double>
typealias Function2 = (x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double>

fun testSigmoid() {
    val a = fill(100, 1) { i, _ -> (i - 50.0) / 5.0 }
    val s = sigmoid(a)
    plot(a,s,"r","sigmoid")
}

fun testSigmoidDerivative() {
    val a = fill(100, 1) { i, _ -> (i - 50.0) / 5.0 }
    val sda = sigmoidDerivative(a)
    plot(a,sda,"g","sigmoid derivative (analytical)")
    val s = sigmoid(a)
    val sdn = s.mapIndexed { r: Int, c: Int, ele: Double ->
        (s[r,c] - s[(r-1).coerceAtLeast(0),c]) / (a[1,c] - a[0, c])
    }
    plot(a,sdn,"b","sigmoid derivative (numerical)")

}

fun testQuadraticCostDerivative() {
    val a = fill(100, 1) { i, _ -> (i - 50.0) / 5.0 }
    val y = zeros(100, 1)
    val s = quadraticCostDerivative(a,y)
    plot(a,s)
}


class Network(
    val sizes: List<Int>,
    val sigma: Function1 = ::sigmoid,
    val sigmaDerivative: Function1 = ::sigmoidDerivative,
    val costDerivative: Function2 = ::quadraticCostDerivative
) {

    private var weights: List<Matrix<Double>> = sizes.zipWithNext { j, k ->
        randn(k,j)
        // each of the j nodes in the previous layer will be connected
        // to all of the k nodes of the following layer
    }

    private var biases: List<Matrix<Double>> = sizes.zipWithNext { _ , k ->
        randn(k, 1)
    }

    fun feedForward(input: Matrix<Double>): Matrix<Double> {

        input.validate { sizes.first() x 1 }

        var a = input

        weights.zip(biases).forEach { (w, b) ->
            a = sigma(w * a + b)
        }

        return a
    }

    fun train(data: List<Case>, batchSize: Int, epochs: Int, learningRate: Double, doAfterEpoch: (Int) -> Unit = {}) {

        require(data.size % batchSize == 0)

        repeat(epochs) {
            //println("Epoch $it:")
            data.shuffled().chunked(batchSize).forEach { batch ->
                updateBatch(batch, learningRate)
            }

            doAfterEpoch(it)
        }

    }

    private fun updateBatch(batch: List<Case>, learningRate: Double) {

        var nablaB = biases.mapTo(arrayListOf()) { zeros(it.numRows(), it.numCols()) }
        var nablaW = weights.mapTo(arrayListOf()) { zeros(it.numRows(), it.numCols()) }

        batch.forEach{ case ->
            val deltas = backPropagate(case)
            nablaB = deltas.bs.zip(nablaB).mapTo(arrayListOf()) { (dnb, nb) ->
                nb + dnb
            }
            nablaW = deltas.ws.zip(nablaW).mapTo(arrayListOf()) { (dnw, nw) ->
                nw + dnw
            }
        }

        weights = weights.zip(nablaW).map { (w, nw) ->
            w - nw * (learningRate / batch.size)
        }

        biases = biases.zip(nablaB).map { (b, nb) ->
            b - nb * (learningRate / batch.size)
        }
    }

    private fun backPropagate(case: Case): Gradients {

        val bGrad = biases.mapTo(arrayListOf()) { zeros(it.numRows(), it.numCols()) }
        val wGrad = weights.mapTo(arrayListOf()) { zeros(it.numRows(), it.numCols()) }

        // Feedforward:

        var a = case.input
        val activations = arrayListOf(a)
        val zs = arrayListOf<Matrix<Double>>()

        weights.zip(biases).forEach { (w, b) ->
            val z = w * a + b
            zs.add(z)
            a = sigmoid(z)
            activations.add(a)
        }

        // Backward pass:

        var delta = costDerivative(activations.last(), case.output) emul sigmaDerivative(zs.last())

        val last = bGrad.lastIndex

        bGrad[last] = delta
        wGrad[last] = delta * activations[last].T

        for (i in last-1 downTo 0) {
            val z = zs[i]
            val sp = sigmaDerivative(z)
            val back = weights[i+1].T * delta
            delta = back emul sp
            bGrad[i] = delta
            wGrad[i] = delta * activations[i].T

        }

        return Gradients(wGrad, bGrad)
    }

    fun test(cases: List<Case>) = cases.filter {
            val res = feedForward(it.input)
            res.argMax() == it.output.argMax()
    }.count()
}