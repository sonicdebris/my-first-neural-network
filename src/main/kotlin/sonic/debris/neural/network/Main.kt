package sonic.debris.neural.network

import koma.cumsum
import koma.randn
import loadDigitImages
import java.io.File

import koma.*
import koma.extensions.*

fun main(args: Array<String>) {
    println("load images...")

    val images = loadDigitImages("train-labels","train-images")
    println("loaded ${images.size}")

    val dest = File(System.getProperty("user.dir"), "images").apply { mkdirs() }
    println("save some files to: ${dest.absolutePath}")

    (0..20).forEach {
        images.random().writeToPng(dest, it.toString())
    }

    val a = randn(100,2)
    val b = cumsum(a)

    figure(1)
    // Second parameter is color
    plot(a, 'b', "First Run")
    plot(a+1, 'y', "First Run Offset")
    xlabel("Time (s)")
    ylabel("Magnitude")
    title("White Noise")

    figure(2)
    plot(b, 'g') // green
    xlabel("Velocity (lightweeks/minute)")
    ylabel("Intelligence")
    title("Random Walk")
}