package sonic.debris.neural.network

import koma.randn
import loadDigitImages
import java.io.File

import koma.*
import koma.extensions.*
import koma.matrix.Matrix

fun main(args: Array<String>) {
    println("load images...")

    val images = loadDigitImages("train-labels","train-images")
    println("loaded ${images.size}")

    val network = Network(listOf(4,3,2))
    val res = network.feedForward(eye(4,1))
    println(res)

    val cases = (0..9).map {
        Case(fill(4,1, it.toDouble()), fill(2,1, it.toDouble()))
    }

    network.train(cases, 2, 3, 0.5)

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