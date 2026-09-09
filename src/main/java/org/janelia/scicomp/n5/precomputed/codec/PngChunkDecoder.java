package org.janelia.scicomp.n5.precomputed.codec;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;

import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.ShortArrayDataBlock;
import org.janelia.saalfeldlab.n5.readdata.ReadData;

/**
 * Decodes a PNG-encoded precomputed chunk.
 * <p>
 * Supports {@code uint8} and {@code uint16} with 1-4 channels. The chunk is a
 * 2D image whose pixels, scanned in row-major order, correspond to the
 * flattened {@code [x, y, z]} Fortran-order chunk (x fastest).
 * <p>
 * For 8-bit multi-channel images the RGBA values are read via
 * {@link BufferedImage#getRGB} (which has a well-defined channel order,
 * unlike the raster band order after ImageIO decoding). Single-channel and
 * 16-bit images are read directly from the raster.
 *
 * @author Stephan Preibisch
 */
public class PngChunkDecoder {

	private PngChunkDecoder() {}

	public static DataBlock<?> decode(
			final ReadData readData,
			final DataType dataType,
			final int[] blockSize,
			final int numChannels,
			final long[] gridPosition) throws N5IOException {

		if (numChannels < 1 || numChannels > 4)
			throw new N5IOException("png encoding only supports 1-4 channels, not " + numChannels);

		final BufferedImage img = JpegChunkDecoder.read(readData);
		final int width = img.getWidth();
		final int spatial = blockSize[0] * blockSize[1] * blockSize[2];

		switch (dataType) {
		case UINT8:
		case INT8: {
			final byte[] data = new byte[spatial * numChannels];
			if (numChannels == 1) {
				final Raster raster = img.getRaster();
				for (int p = 0; p < spatial; ++p)
					data[p] = (byte)raster.getSample(p % width, p / width, 0);
			} else {
				for (int p = 0; p < spatial; ++p) {
					final int argb = img.getRGB(p % width, p / width);
					final int[] comps = {(argb >> 16) & 0xff, (argb >> 8) & 0xff, argb & 0xff, (argb >> 24) & 0xff};
					for (int c = 0; c < numChannels; ++c)
						data[p + spatial * c] = (byte)comps[c];
				}
			}
			return new ByteArrayDataBlock(blockSize, gridPosition, data);
		}
		case UINT16:
		case INT16: {
			final Raster raster = img.getRaster();
			final int numBands = raster.getNumBands();
			final short[] data = new short[spatial * numChannels];
			for (int c = 0; c < numChannels; ++c) {
				final int band = Math.min(c, numBands - 1);
				for (int p = 0; p < spatial; ++p)
					data[p + spatial * c] = (short)raster.getSample(p % width, p / width, band);
			}
			return new ShortArrayDataBlock(blockSize, gridPosition, data);
		}
		default:
			throw new N5IOException("png encoding only supports uint8/uint16, not " + dataType);
		}
	}
}
