package sonic.debris.neural.network

import koma.randn
import DigitImage
import loadDigitImages
import java.io.File
import koma.*
import koma.extensions.*
import koma.matrix.Matrix

fun Byte.toPositiveDouble() = (toInt() and 0xFF).toDouble()

fun DigitImage.toCase(): Case {
    val input = Matrix(data.size, 1) { r, _ -> data[r].toPositiveDouble() / 255.0 }
    val output = Matrix(10, 1) { r, _ -> if (r == label) 1.0 else 0.0 }
    check(output.argMax() == label)
    check(input.numRows() == 784)
    check(input.min() >= 0.0)
    return Case(input, output)
}

fun randomGuess(cases: List<Case>): Int {
    val outs = arrayListOf(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    // guess over the entire set 11 times and return the best success rate
    return (0 to 10).toList().map {
        cases.filter { case ->
            val res = outs.random()
            val actualMax = case.output.argMax()
            res == actualMax
        }.count()
    }.max() ?: -1
}

const val HIDDEN_LAYER_NUM_NEURONS = 30
const val NUM_TRAINING_EPOCHS = 30
const val MINI_BATCH_SIZE = 10
const val LEARNING_RATE = 3.0

fun main(args: Array<String>) {
    println("load images...")

    val images = loadDigitImages("train-labels","train-images")
    val cases = images.map { it.toCase() }
    val testImages = loadDigitImages("test-labels", "test-images")
    val testCases = testImages.map { it.toCase() }
    println("Training over ${cases.size} cases, testing over ${testCases.size} cases.")

    val guessSuccess = randomGuess(testCases)
    println("Guessing - Num. correct answers: $guessSuccess, rate: ${guessSuccess.toFloat()/testCases.size*100}%")

    val inLayerSize = cases[0].input.numRows()
    check(inLayerSize == 784)

    val outLayerSize = cases[0].output.numRows()
    check(outLayerSize == 10)

    val network = Network(listOf(inLayerSize, HIDDEN_LAYER_NUM_NEURONS, outLayerSize))

    network.train(cases, MINI_BATCH_SIZE, NUM_TRAINING_EPOCHS, LEARNING_RATE) {

        val nSuccess = network.test(testCases)
        println("Epoch $it - Num. correct answers: $nSuccess, rate: ${nSuccess.toFloat()/testCases.size * 100}%")

    }

//    val dest = File(System.getProperty("user.dir"), "images").apply { mkdirs() }
//    println("save some files to: ${dest.absolutePath}")
//
//    (0..20).forEach {
//        images.random().writeToPng(dest, it.toString())
//    }


//    val a = randn(100,2)
//    val b = cumsum(a)
//
//    figure(1)
//    // Second parameter is color
//    plot(a, 'b', "First Run")
//    plot(a+1, 'y', "First Run Offset")
//    xlabel("Time (s)")
//    ylabel("Magnitude")
//    title("White Noise")
//
//    figure(2)
//    plot(b, 'g') // green
//    xlabel("Velocity (lightweeks/minute)")
//    ylabel("Intelligence")
//    title("Random Walk")

//    val x = randn(1, 5)
//    println(x)
//
//    val y = mat[1, 2, 3 end 2, 4, 6 end 3, 6, 9]
//    println(y)
//    println(y[1..2, 1..2])
//
//    val z = Matrix(3,3) { row, col -> (col + 1.0)*(row + 1.0) }
//    println(z)
//
//    val w = y * z
//    val ew = y emul z
//    println(w)
//    println(ew)
//
//    val v = mat[1.0, 1.0, 1.0]
//    val r = v * y
//    println(r)
//
//    val v2 = mat[1.0, 2.0, 3.0]
//    val r2 = v * v2.T
//    println(r2)
//
//    val er2 = v emul v2
//    println(er2)



}