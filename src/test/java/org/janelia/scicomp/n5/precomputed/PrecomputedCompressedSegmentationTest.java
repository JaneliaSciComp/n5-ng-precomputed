package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.scicomp.n5.precomputed.PrecomputedTestVolumes.ValueFn;
import org.junit.Test;

public class PrecomputedCompressedSegmentationTest {

	@Test
	public void testUint32() throws IOException {

		// mixes bit depth 0 (uniform) and >0 blocks, and an edge chunk
		final ValueFn fn = (x, y, z, c) -> (x % 2) + 2 * (y % 2) + 4 * (z % 2) + 100L * ((x / 2 + y / 2 + z / 2) % 3);
		roundTrip(DataType.UINT32, 1, new long[]{5, 4, 3}, new int[]{4, 4, 4}, new int[]{2, 2, 2}, fn);
	}

	@Test
	public void testUint64MultiChannel() throws IOException {

		// 64-bit values above 2^32, two channels, edge chunks
		final ValueFn fn = (x, y, z, c) -> ((x * 7 + y * 13 + z * 3) % 5) + (((long)c + 1) << 40);
		roundTrip(DataType.UINT64, 2, new long[]{6, 6, 2}, new int[]{4, 4, 2}, new int[]{2, 2, 2}, fn);
	}

	private void roundTrip(
			final DataType dataType,
			final int numChannels,
			final long[] size,
			final int[] chunk,
			final int[] csBlock,
			final ValueFn fn) throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-cseg");
		final String key = "1_1_1";
		final long[] offset = {0, 0, 0};
		final String extra = "      \"compressed_segmentation_block_size\": ["
				+ csBlock[0] + ", " + csBlock[1] + ", " + csBlock[2] + "],\n";
		PrecomputedTestVolumes.writeInfo(dir, "segmentation", dataType, numChannels, key, size, chunk, offset,
				"compressed_segmentation", extra);
		final Path scaleDir = dir.resolve(key);
		Files.createDirectories(scaleDir);

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (size[d] + chunk[d] - 1) / chunk[d];

		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {
					final long[] g = {gx, gy, gz};
					final long[] begin = new long[3];
					final int[] s = new int[3];
					for (int d = 0; d < 3; ++d) {
						begin[d] = g[d] * chunk[d];
						s[d] = (int)Math.min(chunk[d], size[d] - begin[d]);
					}
					final byte[] bytes = encodeChunk(dataType, numChannels, s, csBlock, begin, fn);
					final String name = begin[0] + "-" + (begin[0] + s[0]) + "_"
							+ begin[1] + "-" + (begin[1] + s[1]) + "_"
							+ begin[2] + "-" + (begin[2] + s[2]);
					Files.write(scaleDir.resolve(name), bytes);
				}

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes(key);
		assertEquals(dataType, attrs.getDataType());
		assertArrayEquals(new long[]{size[0], size[1], size[2], numChannels}, attrs.getDimensions());

		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {
					final DataBlock<?> block = n5.readBlock(key, attrs, gx, gy, gz, 0);
					final long[] g = {gx, gy, gz};
					final long[] begin = new long[3];
					final int[] s = new int[3];
					for (int d = 0; d < 3; ++d) {
						begin[d] = g[d] * chunk[d];
						s[d] = (int)Math.min(chunk[d], size[d] - begin[d]);
					}
					int idx = 0;
					for (int c = 0; c < numChannels; ++c)
						for (int z = 0; z < s[2]; ++z)
							for (int y = 0; y < s[1]; ++y)
								for (int x = 0; x < s[0]; ++x, ++idx) {
									final long expected = fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c);
									final long actual = dataType == DataType.UINT32
											? ((int[])block.getData())[idx] & 0xffffffffL
											: ((long[])block.getData())[idx];
									assertEquals("@ block " + gx + "," + gy + "," + gz + " idx " + idx, expected, actual);
								}
				}
	}

	// -- a spec-compliant compressed_segmentation encoder for testing ---------

	private static byte[] encodeChunk(
			final DataType dataType,
			final int numChannels,
			final int[] s,           // clamped chunk size [x,y,z]
			final int[] b,           // block size [x,y,z]
			final long[] begin,      // absolute voxel origin [x,y,z]
			final ValueFn fn) {

		final boolean is64 = dataType == DataType.UINT64 || dataType == DataType.INT64;
		final int unitsPerElement = is64 ? 2 : 1;

		final int[] grid = new int[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (s[d] + b[d] - 1) / b[d];
		final int numBlocks = grid[0] * grid[1] * grid[2];
		final int blockVoxels = b[0] * b[1] * b[2];

		final List<Integer> words = new ArrayList<>();
		for (int c = 0; c < numChannels; ++c)
			words.add(0); // channel offset table (filled later)

		final int[] channelBase = new int[numChannels];
		for (int c = 0; c < numChannels; ++c) {
			channelBase[c] = words.size();
			final int headerBase = words.size();
			for (int i = 0; i < 2 * numBlocks; ++i)
				words.add(0);

			for (int bidx = 0; bidx < numBlocks; ++bidx) {
				final int bgx = bidx % grid[0];
				final int bgy = (bidx / grid[0]) % grid[1];
				final int bgz = bidx / (grid[0] * grid[1]);

				final LinkedHashMap<Long, Integer> table = new LinkedHashMap<>();
				final int[] indices = new int[blockVoxels];
				for (int zz = 0; zz < b[2]; ++zz)
					for (int yy = 0; yy < b[1]; ++yy)
						for (int xx = 0; xx < b[0]; ++xx) {
							final int p = xx + b[0] * (yy + b[1] * zz);
							final int vx = bgx * b[0] + xx;
							final int vy = bgy * b[1] + yy;
							final int vz = bgz * b[2] + zz;
							if (vx < s[0] && vy < s[1] && vz < s[2]) {
								final long v = fn.apply(begin[0] + vx, begin[1] + vy, begin[2] + vz, c);
								indices[p] = table.computeIfAbsent(v, k -> table.size());
							} else {
								indices[p] = 0;
							}
						}

				final int numDistinct = table.size();
				final int encodingBits = bitsFor(numDistinct);

				int encodedValuesOffset = 0;
				if (encodingBits > 0) {
					encodedValuesOffset = words.size() - channelBase[c];
					final int numEncWords = (blockVoxels * encodingBits + 31) / 32;
					final int[] enc = new int[numEncWords];
					final int mask = (1 << encodingBits) - 1;
					for (int p = 0; p < blockVoxels; ++p) {
						final int bitOffset = encodingBits * p;
						final int word = bitOffset / 32;
						final int shift = bitOffset % 32;
						enc[word] |= (indices[p] & mask) << shift;
					}
					for (final int w : enc)
						words.add(w);
				}

				final int tableOffset = words.size() - channelBase[c];
				for (final Long v : table.keySet()) {
					if (is64) {
						words.add((int)(v & 0xffffffffL));
						words.add((int)(v >>> 32));
					} else {
						words.add((int)(long)v);
					}
				}

				final int h0 = (tableOffset & 0xffffff) | (encodingBits << 24);
				final int h1 = encodingBits > 0 ? encodedValuesOffset : 0;
				words.set(headerBase + bidx * 2, h0);
				words.set(headerBase + bidx * 2 + 1, h1);
			}
		}

		for (int c = 0; c < numChannels; ++c)
			words.set(c, channelBase[c]);

		final ByteBuffer buf = ByteBuffer.allocate(words.size() * 4).order(ByteOrder.LITTLE_ENDIAN);
		for (final int w : words)
			buf.putInt(w);
		return buf.array();
	}

	private static int bitsFor(final int numDistinct) {

		if (numDistinct <= 1)
			return 0;
		final int needed = 32 - Integer.numberOfLeadingZeros(numDistinct - 1);
		int bits = 1;
		while (bits < needed)
			bits *= 2;
		return bits;
	}
}
