package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.zip.GZIPOutputStream;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.scicomp.n5.precomputed.PrecomputedTestVolumes.ValueFn;
import org.janelia.scicomp.n5.precomputed.sharding.CompressedMortonCode;
import org.janelia.scicomp.n5.precomputed.sharding.MurmurHash3;
import org.junit.Test;

public class PrecomputedShardedTest {

	private static final ValueFn FN = (x, y, z, c) -> x + 7 * y + 53 * z + 1009L * c;

	@Test
	public void testMortonCode() {

		final long[] g222 = {2, 2, 2};
		assertEquals(0, CompressedMortonCode.encode(new long[]{0, 0, 0}, g222));
		assertEquals(1, CompressedMortonCode.encode(new long[]{1, 0, 0}, g222));
		assertEquals(2, CompressedMortonCode.encode(new long[]{0, 1, 0}, g222));
		assertEquals(4, CompressedMortonCode.encode(new long[]{0, 0, 1}, g222));
		assertEquals(7, CompressedMortonCode.encode(new long[]{1, 1, 1}, g222));
		// x needs 2 bits, y and z need 0 -> code == x
		assertEquals(3, CompressedMortonCode.encode(new long[]{3, 0, 0}, new long[]{4, 1, 1}));
		// bits [1,2,0]: code = x0 | (y0<<1) | (y1<<2); (1,2,0) -> 1 | 0 | 4 = 5
		assertEquals(5, CompressedMortonCode.encode(new long[]{1, 2, 0}, new long[]{2, 4, 1}));
	}

	@Test
	public void testMurmurEmptyIsZero() {

		assertEquals(0L, MurmurHash3.hashX86_128Low64(new byte[0]));
	}

	@Test
	public void testShardedRawSingleMinishard() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-shard-raw");
		writeSharded(dir, DataType.UINT16, 2, new long[]{5, 4, 3}, new int[]{2, 2, 2},
				0, "identity", 0, "raw", "raw", FN, (x, y, z) -> true);
		verify(dir, DataType.UINT16, 2, new long[]{5, 4, 3}, new int[]{2, 2, 2}, FN, (x, y, z) -> true);
	}

	@Test
	public void testShardedGzipMultiMinishard() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-shard-gzip");
		writeSharded(dir, DataType.UINT16, 1, new long[]{9, 7, 5}, new int[]{2, 2, 2},
				0, "identity", 2, "gzip", "gzip", FN, (x, y, z) -> true);
		verify(dir, DataType.UINT16, 1, new long[]{9, 7, 5}, new int[]{2, 2, 2}, FN, (x, y, z) -> true);
	}

	@Test
	public void testShardedMurmur() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-shard-murmur");
		writeSharded(dir, DataType.UINT16, 1, new long[]{8, 6, 4}, new int[]{2, 2, 2},
				0, "murmurhash3_x86_128", 1, "raw", "gzip", FN, (x, y, z) -> true);
		verify(dir, DataType.UINT16, 1, new long[]{8, 6, 4}, new int[]{2, 2, 2}, FN, (x, y, z) -> true);
	}

	@Test
	public void testShardedMissingChunkReturnsNull() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-shard-missing");
		final long[] size = {6, 4, 2};
		final int[] chunk = {2, 2, 2};
		// omit the chunk at grid (1,0,0)
		final Present present = (x, y, z) -> !(x == 1 && y == 0 && z == 0);
		writeSharded(dir, DataType.UINT16, 1, size, chunk, 0, "identity", 1, "raw", "raw", FN, present);

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertNull(n5.readBlock("1_1_1", attrs, 1, 0, 0, 0));
		assertNotNull(n5.readBlock("1_1_1", attrs, 0, 0, 0, 0));
		verify(dir, DataType.UINT16, 1, size, chunk, FN, present);
	}

	// -- verification ---------------------------------------------------------

	private interface Present {

		boolean at(long gx, long gy, long gz);
	}

	private static void verify(
			final Path dir,
			final DataType dataType,
			final int numChannels,
			final long[] size,
			final int[] chunk,
			final ValueFn fn,
			final Present present) {

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertArrayEquals(new long[]{size[0], size[1], size[2], numChannels}, attrs.getDimensions());

		final long[] grid = grid(size, chunk);
		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {
					final DataBlock<?> block = n5.readBlock("1_1_1", attrs, gx, gy, gz, 0);
					if (!present.at(gx, gy, gz)) {
						assertNull(block);
						continue;
					}
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
									final long expected = fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c) & 0xffff;
									final long actual = ((short[])block.getData())[idx] & 0xffffL;
									assertEquals("@ " + gx + "," + gy + "," + gz + " idx " + idx, expected, actual);
								}
				}
	}

	// -- a single-shard writer (shard_bits = 0) for testing -------------------

	private static void writeSharded(
			final Path dir,
			final DataType dataType,
			final int numChannels,
			final long[] size,
			final int[] chunk,
			final int preshiftBits,
			final String hashName,
			final int minishardBits,
			final String minishardIndexEncoding,
			final String dataEncoding,
			final ValueFn fn,
			final Present present) {

		try {
			final String key = "1_1_1";
			final long[] offset = {0, 0, 0};
			final String sharding = "      \"sharding\": {"
					+ "\"@type\": \"neuroglancer_uint64_sharded_v1\", "
					+ "\"preshift_bits\": " + preshiftBits + ", "
					+ "\"hash\": \"" + hashName + "\", "
					+ "\"minishard_bits\": " + minishardBits + ", "
					+ "\"shard_bits\": 0, "
					+ "\"minishard_index_encoding\": \"" + minishardIndexEncoding + "\", "
					+ "\"data_encoding\": \"" + dataEncoding + "\"},\n";
			PrecomputedTestVolumes.writeInfo(dir, "image", dataType, numChannels, key, size, chunk, offset, "raw", sharding);

			final Path scaleDir = dir.resolve(key);
			Files.createDirectories(scaleDir);

			final long[] grid = grid(size, chunk);
			final long[] gridSize = {grid[0], grid[1], grid[2]};

			final List<Chunk> chunks = new ArrayList<>();
			for (long gz = 0; gz < grid[2]; ++gz)
				for (long gy = 0; gy < grid[1]; ++gy)
					for (long gx = 0; gx < grid[0]; ++gx) {
						if (!present.at(gx, gy, gz))
							continue;
						final long[] g = {gx, gy, gz};
						final long[] begin = new long[3];
						final int[] s = new int[3];
						for (int d = 0; d < 3; ++d) {
							begin[d] = g[d] * chunk[d];
							s[d] = (int)Math.min(chunk[d], size[d] - begin[d]);
						}
						final long morton = CompressedMortonCode.encode(g, gridSize);
						final byte[] raw = PrecomputedTestVolumes.rawChunkBytes(dataType, numChannels, s, begin, fn);
						chunks.add(new Chunk(morton, dataEncoding.equals("gzip") ? gzip(raw) : raw));
					}

			final byte[] shardBytes = buildShard(chunks, preshiftBits, hashName, minishardBits, minishardIndexEncoding);
			Files.write(scaleDir.resolve("0.shard"), shardBytes);
		} catch (final IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	private static byte[] buildShard(
			final List<Chunk> chunks,
			final int preshiftBits,
			final String hashName,
			final int minishardBits,
			final String minishardIndexEncoding) throws IOException {

		final int numMinishards = 1 << minishardBits;
		final long minishardMask = numMinishards - 1L;

		final List<List<Chunk>> groups = new ArrayList<>();
		for (int i = 0; i < numMinishards; ++i)
			groups.add(new ArrayList<>());
		for (final Chunk chunk : chunks) {
			final long hashed = hash(hashName, chunk.morton >>> preshiftBits);
			final int ms = (int)(hashed & minishardMask);
			groups.get(ms).add(chunk);
		}
		for (final List<Chunk> group : groups)
			group.sort(Comparator.comparingLong(a -> a.morton));

		// data region (contiguous, in minishard then morton order)
		final ByteArrayOutputStream data = new ByteArrayOutputStream();
		for (final List<Chunk> group : groups)
			for (final Chunk chunk : group) {
				chunk.relOffset = data.size();
				data.write(chunk.encoded);
			}
		final long totalData = data.size();

		// minishard index region + shard index entries
		final ByteArrayOutputStream miRegion = new ByteArrayOutputStream();
		final long[] miStart = new long[numMinishards];
		final long[] miEnd = new long[numMinishards];
		long cursor = totalData;
		for (int ms = 0; ms < numMinishards; ++ms) {
			final List<Chunk> group = groups.get(ms);
			if (group.isEmpty()) {
				miStart[ms] = cursor;
				miEnd[ms] = cursor;
				continue;
			}
			final int n = group.size();
			final long[] vals = new long[3 * n];
			long prevId = 0;
			for (int i = 0; i < n; ++i) {
				vals[i] = group.get(i).morton - prevId;
				prevId = group.get(i).morton;
			}
			long prevEnd = 0;
			for (int i = 0; i < n; ++i) {
				final long off = group.get(i).relOffset;
				vals[n + i] = off - prevEnd;
				vals[2 * n + i] = group.get(i).encoded.length;
				prevEnd = off + group.get(i).encoded.length;
			}
			byte[] enc = longsToLE(vals);
			if (minishardIndexEncoding.equals("gzip"))
				enc = gzip(enc);
			miStart[ms] = cursor;
			miEnd[ms] = cursor + enc.length;
			cursor += enc.length;
			miRegion.write(enc);
		}

		final ByteBuffer header = ByteBuffer.allocate(numMinishards * 16).order(ByteOrder.LITTLE_ENDIAN);
		for (int ms = 0; ms < numMinishards; ++ms) {
			header.putLong(miStart[ms]);
			header.putLong(miEnd[ms]);
		}

		final ByteArrayOutputStream out = new ByteArrayOutputStream();
		out.write(header.array());
		out.write(data.toByteArray());
		out.write(miRegion.toByteArray());
		return out.toByteArray();
	}

	private static long hash(final String hashName, final long value) {

		return hashName.equals("identity") ? value : MurmurHash3.hashX86_128Low64(value);
	}

	private static long[] grid(final long[] size, final int[] chunk) {

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (size[d] + chunk[d] - 1) / chunk[d];
		return grid;
	}

	private static byte[] longsToLE(final long[] vals) {

		final ByteBuffer bb = ByteBuffer.allocate(vals.length * 8).order(ByteOrder.LITTLE_ENDIAN);
		for (final long v : vals)
			bb.putLong(v);
		return bb.array();
	}

	private static byte[] gzip(final byte[] bytes) throws IOException {

		final ByteArrayOutputStream out = new ByteArrayOutputStream();
		try (final GZIPOutputStream gz = new GZIPOutputStream(out)) {
			gz.write(bytes);
		}
		return out.toByteArray();
	}

	private static final class Chunk {

		final long morton;
		final byte[] encoded;
		int relOffset;

		Chunk(final long morton, final byte[] encoded) {

			this.morton = morton;
			this.encoded = encoded;
		}
	}
}
