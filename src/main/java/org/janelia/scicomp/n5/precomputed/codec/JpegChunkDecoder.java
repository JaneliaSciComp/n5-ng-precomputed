package org.janelia.scicomp.n5.precomputed.codec;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.readdata.ReadData;

/**
 * Decodes a JPEG-encoded precomputed chunk.
 * <p>
 * Supports {@code uint8} with 1 or 3 channels. The chunk is a 2D image whose
 * pixels, scanned in row-major order, correspond to the flattened
 * {@code [x, y, z]} Fortran-order chunk (x fastest). For 3-channel (RGB)
 * images the decoded RGB values are used (rather than the raw raster, which may
 * be YCbCr).
 *
 * @author Stephan Preibisch
 */
public class JpegChunkDecoder {

	private JpegChunkDecoder() {}

	public static DataBlock<byte[]> decode(
			final ReadData readData,
			final DataType dataType,
			final int[] blockSize,
			final int numChannels,
			final long[] gridPosition) throws N5IOException {

		if (dataType != DataType.UINT8 && dataType != DataType.INT8)
			throw new N5IOException("jpeg encoding only supports uint8/int8, not " + dataType);
		if (numChannels != 1 && numChannels != 3)
			throw new N5IOException("jpeg encoding only supports 1 or 3 channels, not " + numChannels);

		final BufferedImage img = read(readData);
		final int width = img.getWidth();
		final int spatial = blockSize[0] * blockSize[1] * blockSize[2];

		final byte[] data = new byte[spatial * numChannels];
		if (numChannels == 1) {
			final Raster raster = img.getRaster();
			for (int p = 0; p < spatial; ++p)
				data[p] = (byte)raster.getSample(p % width, p / width, 0);
		} else {
			for (int p = 0; p < spatial; ++p) {
				final int rgb = img.getRGB(p % width, p / width);
				data[p] = (byte)((rgb >> 16) & 0xff);
				data[p + spatial] = (byte)((rgb >> 8) & 0xff);
				data[p + 2 * spatial] = (byte)(rgb & 0xff);
			}
		}
		return new ByteArrayDataBlock(blockSize, gridPosition, data);
	}

	static BufferedImage read(final ReadData readData) throws N5IOException {

		try {
			final BufferedImage img = ImageIO.read(new ByteArrayInputStream(readData.allBytes()));
			if (img == null)
				throw new N5IOException("could not decode image chunk (no suitable ImageIO reader)");
			return img;
		} catch (final IOException e) {
			throw new N5IOException("failed to decode image chunk", e);
		}
	}
}
