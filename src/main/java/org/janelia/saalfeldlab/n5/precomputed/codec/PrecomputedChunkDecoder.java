package org.janelia.saalfeldlab.n5.precomputed.codec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.janelia.saalfeldlab.n5.ByteArrayDataBlock;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DoubleArrayDataBlock;
import org.janelia.saalfeldlab.n5.FloatArrayDataBlock;
import org.janelia.saalfeldlab.n5.IntArrayDataBlock;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.ShortArrayDataBlock;

/**
 * Decodes the raw bytes of a Neuroglancer precomputed chunk into a
 * {@link DataBlock}, dispatching on the scale's {@code encoding}.
 * <p>
 * (This replaces the n5-4.x {@code BlockCodec} pipeline, which does not exist in
 * n5&nbsp;3.x.) {@code blockSize} is the clamped size of this chunk
 * ({@code [x, y, z, channel]}) — precomputed stores edge chunks at their true,
 * smaller extent.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedChunkDecoder {

	public static final String ENCODING_RAW = "raw";
	public static final String ENCODING_JPEG = "jpeg";
	public static final String ENCODING_PNG = "png";
	public static final String ENCODING_COMPRESSED_SEGMENTATION = "compressed_segmentation";

	private PrecomputedChunkDecoder() {}

	public static DataBlock<?> decode(
			final String encoding,
			final DataType dataType,
			final int[] blockSize,
			final int numChannels,
			final int[] compressedSegmentationBlockSize,
			final long[] gridPosition,
			final byte[] bytes) throws N5IOException {

		switch (encoding) {
		case ENCODING_RAW:
			return raw(dataType, blockSize, gridPosition, bytes);
		case ENCODING_JPEG:
			return JpegChunkDecoder.decode(bytes, dataType, blockSize, numChannels, gridPosition);
		case ENCODING_PNG:
			return PngChunkDecoder.decode(bytes, dataType, blockSize, numChannels, gridPosition);
		case ENCODING_COMPRESSED_SEGMENTATION:
			return CompressedSegmentationDecoder.decode(
					bytes, dataType, blockSize, numChannels, compressedSegmentationBlockSize, gridPosition);
		default:
			throw new N5Exception("Unsupported precomputed encoding: " + encoding);
		}
	}

	/** Decodes a headerless little-endian raw chunk. */
	private static DataBlock<?> raw(
			final DataType dataType,
			final int[] blockSize,
			final long[] gridPosition,
			final byte[] bytes) {

		int n = 1;
		for (final int s : blockSize)
			n *= s;
		final ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

		switch (dataType) {
		case UINT8:
		case INT8: {
			final byte[] d = new byte[n];
			bb.get(d);
			return new ByteArrayDataBlock(blockSize, gridPosition, d);
		}
		case UINT16:
		case INT16: {
			final short[] d = new short[n];
			for (int i = 0; i < n; ++i)
				d[i] = bb.getShort();
			return new ShortArrayDataBlock(blockSize, gridPosition, d);
		}
		case UINT32:
		case INT32: {
			final int[] d = new int[n];
			for (int i = 0; i < n; ++i)
				d[i] = bb.getInt();
			return new IntArrayDataBlock(blockSize, gridPosition, d);
		}
		case UINT64:
		case INT64: {
			final long[] d = new long[n];
			for (int i = 0; i < n; ++i)
				d[i] = bb.getLong();
			return new LongArrayDataBlock(blockSize, gridPosition, d);
		}
		case FLOAT32: {
			final float[] d = new float[n];
			for (int i = 0; i < n; ++i)
				d[i] = bb.getFloat();
			return new FloatArrayDataBlock(blockSize, gridPosition, d);
		}
		case FLOAT64: {
			final double[] d = new double[n];
			for (int i = 0; i < n; ++i)
				d[i] = bb.getDouble();
			return new DoubleArrayDataBlock(blockSize, gridPosition, d);
		}
		default:
			throw new N5Exception("Unsupported raw data type: " + dataType);
		}
	}
}
