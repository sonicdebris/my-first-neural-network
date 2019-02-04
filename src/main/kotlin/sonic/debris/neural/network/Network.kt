package sonic.debris.neural.network

import koma.exp
import koma.fill
import koma.matrix.Matrix
import koma.plot
import koma.randn
import koma.util.validation.validate

fun sigmoid(x: Matrix<Double>): Matrix<Double> =  (exp(-x) + 1.0).epow(-1)

fun testSigmoid() {
    val a = fill(100, 1) { i, _ -> (i - 50.0) / 5.0 }
    val s = sigmoid(a)
    plot(a,s)
}

class Network(val sizes: List<Int>) {

    private val weights: List<Matrix<Double>> = sizes.zipWithNext { j, k ->
        randn(k,j)
        // each of the j nodes in the previous layer will be connected
        // to all of the k nodes of the following layer
    }

    private val biases: List<Matrix<Double>> = sizes.zipWithNext { _ , k ->
        randn(k, 1)
    }

    fun feedForward(input: Matrix<Double>): Matrix<Double> {

        input.validate { sizes.first() x 1 }

        var a = input

        weights.zip(biases).forEach { (w, b) ->
            a = sigmoid(w * a + b)
        }

        return a
    }
}