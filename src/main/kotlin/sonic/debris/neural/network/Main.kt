package sonic.debris.neural.network

import loadDigitImages
import java.io.File

fun main(args: Array<String>) {
    println("load images...")

    val images = loadDigitImages("train-labels","train-images")
    println("loaded ${images.size}")

    val dest = File(System.getProperty("user.dir"), "images").apply { mkdirs() }
    println("save some files to: ${dest.absolutePath}")

    (0..20).forEach {
        images.random().writeToPng(dest, it.toString())
    }
}