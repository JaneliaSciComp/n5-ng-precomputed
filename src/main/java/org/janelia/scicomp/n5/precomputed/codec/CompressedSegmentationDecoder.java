package org.janelia.scicomp.n5.precomputed.codec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.IntArrayDataBlock;
import org.janelia.saalfeldlab.n5.LongArrayDataBlock;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.readdata.ReadData;

/**
 * Decodes a Neuroglancer {@code compressed_segmentation} chunk (uint32/uint64).
 * <p>
 * Ported from the reference decoder
 * (<a href="https://github.com/google/neuroglancer/blob/master/src/sliceview/compressed_segmentation/README.md">format
 * spec</a>). The chunk is decoded at its (clamped) data size; the block grid is
 * {@code ceil(chunkSize / compressed_segmentation_block_size)}. Multiple
 * channels are stored back-to-back preceded by a channel offset table.
 *
 * @author Stephan Preibisch
 */
public class CompressedSegmentationDecoder {

	private CompressedSegmentationDecoder() {}

	public static DataBlock<?> decode(
			final ReadData readData,
			final DataType dataType,
			final int[] blockSize,       // clamped [x, y, z, channel]
			final int numChannels,
			final int[] compressedSegmentationBlockSize, // [x, y, z]
			final long[] gridPosition) throws N5IOException {

		if (compressedSegmentationBlockSize == null)
			throw new N5IOException("compressed_segmentation requires compressed_segmentation_block_size");

		final int[] data = toUint32LE(readData.allBytes());
		final int[] chunkDataSize = {blockSize[0], blockSize[1], blockSize[2]};
		final int channelLen = chunkDataSize[0] * chunkDataSize[1] * chunkDataSize[2];

		switch (dataType) {
		case UINT32:
		case INT32: {
			final int[] out = new int[channelLen * numChannels];
			for (int c = 0; c < numChannels; ++c)
				decodeChannel32(out, channelLen * c, data, data[c], chunkDataSize, compressedSegmentationBlockSize);
			return new IntArrayDataBlock(blockSize, gridPosition, out);
		}
		case UINT64:
		case INT64: {
			final long[] out = new long[channelLen * numChannels];
			for (int c = 0; c < numChannels; ++c)
				decodeChannel64(out, channelLen * c, data, data[c], chunkDataSize, compressedSegmentationBlockSize);
			return new LongArrayDataBlock(blockSize, gridPosition, out);
		}
		default:
			throw new N5IOException("compressed_segmentation only supports uint32/uint64, not " + dataType);
		}
	}

	private static void decodeChannel32(
			final int[] out, final int outBase,
			final int[] data, final int baseOffset,
			final int[] chunkDataSize, final int[] blockSize) {

		final int vx = chunkDataSize[0], vy = chunkDataSize[1], vz = chunkDataSize[2];
		int o = outBase;
		final int[] pos = new int[3];
		for (int z = 0; z < vz; ++z) {
			pos[2] = z;
			for (int y = 0; y < vy; ++y) {
				pos[1] = y;
				for (int x = 0; x < vx; ++x) {
					pos[0] = x;
					final int off = baseOffset + decodeValueOffset(data, baseOffset, chunkDataSize, blockSize, pos, 1);
					out[o++] = data[off];
				}
			}
		}
	}

	private static void decodeChannel64(
			final long[] out, final int outBase,
			final int[] data, final int baseOffset,
			final int[] chunkDataSize, final int[] blockSize) {

		final int vx = chunkDataSize[0], vy = chunkDataSize[1], vz = chunkDataSize[2];
		int o = outBase;
		final int[] pos = new int[3];
		for (int z = 0; z < vz; ++z) {
			pos[2] = z;
			for (int y = 0; y < vy; ++y) {
				pos[1] = y;
				for (int x = 0; x < vx; ++x) {
					pos[0] = x;
					final int off = baseOffset + decodeValueOffset(data, baseOffset, chunkDataSize, blockSize, pos, 2);
					out[o++] = Integer.toUnsignedLong(data[off]) | (Integer.toUnsignedLong(data[off + 1]) << 32);
				}
			}
		}
	}

	/**
	 * Returns the offset (relative to {@code baseOffset}) into {@code data} at
	 * which the value for {@code pos} is stored in the block's lookup table.
	 * Faithful port of {@code decodeValueOffset} in the reference decoder.
	 */
	private static int decodeValueOffset(
			final int[] data, final int baseOffset,
			final int[] chunkDataSize, final int[] blockSize,
			final int[] pos, final int uint32sPerElement) {

		int gridOffset = 0;
		int subchunkOffset = 0;
		int gridStride = 1;
		int subchunkStride = 1;
		for (int i = 0; i < 3; ++i) {
			final int posValue = pos[i];
			final int subSize = blockSize[i];
			gridOffset += (posValue / subSize) * gridStride;
			gridStride *= ceilDiv(chunkDataSize[i], subSize);
			subchunkOffset += (posValue % subSize) * subchunkStride;
			subchunkStride *= subSize;
		}

		final int subchunkHeaderOffset = baseOffset + gridOffset * 2;
		final int header0 = data[subchunkHeaderOffset];
		final int header1 = data[subchunkHeaderOffset + 1];

		int outputValueOffset = header0 & 0xffffff;
		final int encodingBits = (header0 >>> 24) & 0xff;
		if (encodingBits > 0) {
			final int encodedValueBaseOffset = (baseOffset + header1) & 0xffffff;
			final int encodedValueOffset = encodedValueBaseOffset + (subchunkOffset * encodingBits) / 32;
			final int encodedValue = data[encodedValueOffset];
			final int wordOffset = (subchunkOffset * encodingBits) % 32;
			final long mask = encodingBits == 32 ? 0xffffffffL : ((1L << encodingBits) - 1L);
			final long decodedValue = (Integer.toUnsignedLong(encodedValue) >>> wordOffset) & mask;
			outputValueOffset += (int)(uint32sPerElement * decodedValue);
		}
		return outputValueOffset;
	}

	private static int ceilDiv(final int a, final int b) {

		return (a + b - 1) / b;
	}

	private static int[] toUint32LE(final byte[] bytes) {

		final int n = bytes.length / 4;
		final int[] data = new int[n];
		final ByteBuffer bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
		for (int i = 0; i < n; ++i)
			data[i] = bb.getInt();
		return data;
	}
}
