
import java.util.Arrays
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.ArrayList
import java.nio.ByteBuffer
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.image.DataBuffer
import java.awt.Transparency
import java.awt.color.ColorSpace
import java.awt.image.ColorModel
import java.awt.image.ComponentColorModel
import java.awt.image.Raster
import java.awt.image.WritableRaster
import java.awt.image.DataBufferByte



private const val MAGIC_OFFSET = 0
private const val MAGIC_SIZE = 4 //in bytes

private const val LABEL_MAGIC = 2049
private const val IMAGE_MAGIC = 2051

private const val N_ITEMS_OFFSET = 4
private const val N_ITEMS_SIZE = 4

private const val N_ROWS_OFFSET = 8
private const val N_ROWS_SIZE = 4
private const val N_ROWS_EXPECTED = 28

private const val N_COLUMNS_OFFSET = 12
private const val N_COLUMNS_SIZE = 4
private const val N_COLUMNS_EXPECTED = 28

private const val IMAGE_OFFSET = 16
private const val IMAGE_SIZE = N_ROWS_EXPECTED * N_COLUMNS_EXPECTED

class DigitImage(val data: ByteArray, val label: Int) {

    fun writeToPng(destDir: File, prefix: String) {

        val file = File(destDir, "img_${prefix}_$label.png")

        val buffer = DataBufferByte(data, data.size)

        val raster = Raster.createInterleavedRaster(buffer, 28, 28, 28, 1, intArrayOf(0), null)
        val cm = ComponentColorModel(
            ColorSpace.getInstance(ColorSpace.CS_GRAY),
            false,
            true,
            Transparency.OPAQUE,
            DataBuffer.TYPE_BYTE
        )

        val image = BufferedImage(cm, raster, true, null)
        ImageIO.write(image, "png", file)
    }

}

fun loadDigitImages(labelFileName: String, imageFileName: String): List<DigitImage> {

    val images = ArrayList<DigitImage>()

    val labelBuffer = ByteArrayOutputStream()
    val imageBuffer = ByteArrayOutputStream()

    val labelInputStream = DigitImage::class.java.classLoader.getResourceAsStream(labelFileName)
    val imageInputStream = DigitImage::class.java.classLoader.getResourceAsStream(imageFileName)

    check(labelInputStream != null) {"Error loading labels file"}
    check(imageInputStream != null) {"Error loading image file"}

    var read = 0
    val buffer = ByteArray(16384)

    while (read != -1) {
        labelBuffer.write(buffer, 0, read)
        read = labelInputStream.read(buffer, 0, buffer.size)
    }

    labelBuffer.flush()

    read = 0
    while ( read != -1) {
        imageBuffer.write(buffer, 0, read)
        read = imageInputStream.read(buffer, 0, buffer.size)
    }

    imageBuffer.flush()

    val labelBytes = labelBuffer.toByteArray()
    val imageBytes = imageBuffer.toByteArray()

    val labelMagic = Arrays.copyOfRange(labelBytes, MAGIC_OFFSET, MAGIC_SIZE)
    val imageMagic = Arrays.copyOfRange(imageBytes, MAGIC_OFFSET, MAGIC_SIZE)

    if (ByteBuffer.wrap(labelMagic).int != LABEL_MAGIC) {
        error("Load MNIST: wrong labels magic number")
    }

    if (ByteBuffer.wrap(imageMagic).int != IMAGE_MAGIC) {
        error("Load MNIST: wrong image magic number")
    }

    val numberOfLabels = ByteBuffer.wrap(
        Arrays.copyOfRange(labelBytes, N_ITEMS_OFFSET, N_ITEMS_OFFSET + N_ITEMS_SIZE)
    ).int

    val numberOfImages = ByteBuffer.wrap(
        Arrays.copyOfRange(imageBytes, N_ITEMS_OFFSET, N_ITEMS_OFFSET + N_ITEMS_SIZE)
    ).int

    if (numberOfImages != numberOfLabels) {
        error("Load MNIST: number of labels and images don't match")
    }

    val numRows = ByteBuffer.wrap(
        Arrays.copyOfRange(imageBytes, N_ROWS_OFFSET, N_ROWS_OFFSET + N_ROWS_SIZE)
    ).int

    val numCols = ByteBuffer.wrap(
        Arrays.copyOfRange(imageBytes, N_COLUMNS_OFFSET, N_COLUMNS_OFFSET + N_COLUMNS_SIZE)
    ).int

    if (numRows != N_ROWS_EXPECTED || numCols != N_COLUMNS_EXPECTED) {
        error("Load MNIST: bad image. Rows ($numRows) or columns ($numCols) do not equal expected")
    }

    for (i in 0 until numberOfLabels) {

        val label = labelBytes[MAGIC_SIZE + N_ITEMS_SIZE + i].toInt()

        val imageData = Arrays.copyOfRange(
            imageBytes,
            i * IMAGE_SIZE + IMAGE_OFFSET,
            i * IMAGE_SIZE + IMAGE_OFFSET + IMAGE_SIZE
        )

        images.add(DigitImage(imageData, label))
    }

    return images
}