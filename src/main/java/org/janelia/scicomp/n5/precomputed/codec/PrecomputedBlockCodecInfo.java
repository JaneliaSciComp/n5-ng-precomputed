package org.janelia.scicomp.n5.precomputed.codec;

import java.nio.ByteOrder;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.codec.BlockCodec;
import org.janelia.saalfeldlab.n5.codec.BlockCodecInfo;
import org.janelia.saalfeldlab.n5.codec.DataCodecInfo;
import org.janelia.saalfeldlab.n5.codec.IdentityCodec;
import org.janelia.saalfeldlab.n5.codec.RawBlockCodecs;
import org.janelia.saalfeldlab.n5.readdata.ReadData;

/**
 * {@link BlockCodecInfo} for Neuroglancer precomputed chunk encodings.
 * <p>
 * A single {@code PrecomputedBlockCodecInfo} carries the scale's
 * {@code encoding} and everything a decoder needs, and produces a
 * {@link BlockCodec} that:
 * <ul>
 * <li>computes the <em>clamped</em> block size for each grid position (edge
 * chunks in precomputed are physically smaller, not padded);</li>
 * <li>decodes the raw chunk bytes according to the encoding
 * ({@code raw, jpeg, png, compressed_segmentation}), little-endian.</li>
 * </ul>
 * This codec only decodes; encoding is unsupported (the reader is read-only).
 *
 * @author Stephan Preibisch
 */
public class PrecomputedBlockCodecInfo implements BlockCodecInfo {

	private static final long serialVersionUID = 1L;

	public static final String TYPE = "precomputed";

	public static final String ENCODING_RAW = "raw";
	public static final String ENCODING_JPEG = "jpeg";
	public static final String ENCODING_PNG = "png";
	public static final String ENCODING_COMPRESSED_SEGMENTATION = "compressed_segmentation";

	private final String encoding;
	private final long[] dimensions;
	private final int numChannels;
	private final int[] compressedSegmentationBlockSize;

	/**
	 * @param encoding                        the precomputed chunk encoding
	 * @param dimensions                      the dataset dimensions {@code [x,y,z,channel]} (used to clamp edge chunks)
	 * @param numChannels                     the number of channels
	 * @param compressedSegmentationBlockSize the compressed_segmentation block size {@code [x,y,z]} (may be null)
	 */
	public PrecomputedBlockCodecInfo(
			final String encoding,
			final long[] dimensions,
			final int numChannels,
			final int[] compressedSegmentationBlockSize) {

		this.encoding = encoding;
		this.dimensions = dimensions;
		this.numChannels = numChannels;
		this.compressedSegmentationBlockSize = compressedSegmentationBlockSize;
	}

	@Override
	public String getType() {

		return TYPE;
	}

	public String getEncoding() {

		return encoding;
	}

	@Override
	public <T> BlockCodec<T> create(final DataType dataType, final int[] blockSize, final DataCodecInfo... codecs) {

		return new PrecomputedBlockCodec<>(dataType, blockSize);
	}

	/**
	 * Computes the clamped block size for a grid position. Precomputed stores
	 * edge chunks at their true (clamped) extent rather than padded to the full
	 * chunk size; the channel axis is never clamped.
	 *
	 * @param gridPosition   the block grid position
	 * @param fullBlockSize  the nominal (full) block size
	 * @return the clamped block size
	 */
	int[] clampedBlockSize(final long[] gridPosition, final int[] fullBlockSize) {

		final int n = fullBlockSize.length;
		final int[] clamped = new int[n];
		for (int d = 0; d < n; ++d) {
			final long start = gridPosition[d] * (long)fullBlockSize[d];
			final long remain = dimensions[d] - start;
			clamped[d] = (int)Math.min(fullBlockSize[d], Math.max(0, remain));
		}
		return clamped;
	}

	private class PrecomputedBlockCodec<T> implements BlockCodec<T> {

		private final DataType dataType;
		private final int[] fullBlockSize;

		PrecomputedBlockCodec(final DataType dataType, final int[] fullBlockSize) {

			this.dataType = dataType;
			this.fullBlockSize = fullBlockSize;
		}

		@Override
		public ReadData encode(final DataBlock<T> dataBlock) throws N5IOException {

			throw new UnsupportedOperationException("n5-ng-precomputed is read-only");
		}

		@Override
		@SuppressWarnings("unchecked")
		public DataBlock<T> decode(final ReadData readData, final long[] gridPosition) throws N5IOException {

			final int[] blockSize = clampedBlockSize(gridPosition, fullBlockSize);

			switch (encoding) {
			case ENCODING_RAW:
				final BlockCodec<T> raw = RawBlockCodecs.create(dataType, ByteOrder.LITTLE_ENDIAN, blockSize, new IdentityCodec());
				return raw.decode(readData, gridPosition);
			case ENCODING_JPEG:
				return (DataBlock<T>)JpegChunkDecoder.decode(readData, dataType, blockSize, numChannels, gridPosition);
			case ENCODING_PNG:
				return (DataBlock<T>)PngChunkDecoder.decode(readData, dataType, blockSize, numChannels, gridPosition);
			case ENCODING_COMPRESSED_SEGMENTATION:
				return (DataBlock<T>)CompressedSegmentationDecoder.decode(
						readData, dataType, blockSize, numChannels, compressedSegmentationBlockSize, gridPosition);
			default:
				throw new N5Exception("Unsupported precomputed encoding: " + encoding);
			}
		}
	}
}
