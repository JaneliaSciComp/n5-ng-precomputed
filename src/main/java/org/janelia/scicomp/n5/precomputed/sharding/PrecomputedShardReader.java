package org.janelia.scicomp.n5.precomputed.sharding;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPInputStream;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.KeyValueAccess;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.codec.BlockCodec;
import org.janelia.scicomp.n5.precomputed.PrecomputedDatasetAttributes;
import org.janelia.scicomp.n5.precomputed.PrecomputedInfo.Sharding;
import org.janelia.saalfeldlab.n5.readdata.ReadData;
import org.janelia.saalfeldlab.n5.readdata.VolatileReadData;

/**
 * Reads chunks from a Neuroglancer precomputed <em>sharded</em> scale.
 * <p>
 * Given a chunk grid position, this computes the compressed Morton code, maps
 * it to a shard + minishard using the {@link Sharding} parameters, then uses
 * byte-range reads on the shard file (via {@link ReadData#slice}) to read the
 * shard index entry, the minishard index, and finally the chunk bytes — so a
 * single chunk read does not download the whole shard from a cloud backend.
 * The chunk bytes are then decoded with the scale's encoding block codec.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedShardReader {

	private PrecomputedShardReader() {}

	public static <T> DataBlock<T> readBlock(
			final KeyValueAccess kva,
			final URI uri,
			final String normalPath,
			final PrecomputedDatasetAttributes attrs,
			final long[] gridPosition) throws N5Exception {

		final Sharding sh = attrs.getSharding();
		final long morton = CompressedMortonCode.encode(gridPosition, attrs.getGridSize());
		final long preshifted = morton >>> sh.preshiftBits;
		final long hashed = hash(sh.hash, preshifted);

		final long minishard = hashed & mask(sh.minishardBits);
		final long shard = (hashed >>> sh.minishardBits) & mask(sh.shardBits);

		final String shardFile = hex(shard, sh.shardBits) + ".shard";
		final String shardPath = kva.compose(uri, normalPath, shardFile);
		if (!kva.isFile(shardPath))
			return null;

		final long shardIndexEnd = (1L << sh.minishardBits) * 16L;

		// shard index entry for this minishard: 16 bytes (start, end) relative to shardIndexEnd
		final ByteBuffer entry = le(readRange(kva, shardPath, minishard * 16L, 16));
		final long miStart = entry.getLong(0);
		final long miEnd = entry.getLong(8);
		if (miStart == miEnd)
			return null; // empty minishard

		// minishard index: 3n uint64le, optionally gzip-compressed
		final byte[] miRaw = readRange(kva, shardPath, shardIndexEnd + miStart, miEnd - miStart);
		final ByteBuffer mi = le(decode(miRaw, sh.minishardIndexEncoding));
		final int total = mi.capacity() / 8;
		final int n = total / 3;

		// row 0: delta-encoded chunk ids; find target
		long id = 0;
		int target = -1;
		for (int i = 0; i < n; ++i) {
			id += mi.getLong(i * 8);
			if (id == morton) {
				target = i;
				break;
			}
		}
		if (target < 0)
			return null;

		// rows 1 (offset deltas) and 2 (sizes): reconstruct byte range of target chunk
		long prevEnd = 0;
		long chunkOffset = 0;
		long chunkSize = 0;
		for (int i = 0; i <= target; ++i) {
			final long offsetDelta = mi.getLong((n + i) * 8);
			final long size = mi.getLong((2 * n + i) * 8);
			final long offset = prevEnd + offsetDelta;
			prevEnd = offset + size;
			if (i == target) {
				chunkOffset = offset;
				chunkSize = size;
			}
		}

		final byte[] chunkRaw = readRange(kva, shardPath, shardIndexEnd + chunkOffset, chunkSize);
		final byte[] chunkBytes = decode(chunkRaw, sh.dataEncoding);

		final BlockCodec<T> codec = attrs.getBlockCodecInfo().create(attrs.getDataType(), attrs.getBlockSize());
		return codec.decode(ReadData.from(chunkBytes), gridPosition);
	}

	private static long hash(final String hash, final long value) {

		if (hash == null || hash.equals("identity"))
			return value;
		if (hash.equals("murmurhash3_x86_128"))
			return MurmurHash3.hashX86_128Low64(value);
		throw new N5Exception("unsupported sharding hash: " + hash);
	}

	private static long mask(final int bits) {

		return bits >= 64 ? -1L : (bits <= 0 ? 0L : ((1L << bits) - 1L));
	}

	private static String hex(final long shard, final int shardBits) {

		final int width = (shardBits + 3) / 4;
		final String s = Long.toHexString(shard);
		if (s.length() >= width)
			return s;
		final StringBuilder sb = new StringBuilder();
		for (int i = s.length(); i < width; ++i)
			sb.append('0');
		return sb.append(s).toString();
	}

	private static byte[] readRange(final KeyValueAccess kva, final String path, final long offset, final long length)
			throws N5IOException {

		try (final VolatileReadData rd = kva.createReadData(path)) {
			return rd.slice(offset, length).allBytes();
		}
	}

	private static ByteBuffer le(final byte[] bytes) {

		return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
	}

	private static byte[] decode(final byte[] bytes, final String encoding) throws N5IOException {

		if (encoding == null || encoding.equals("raw"))
			return bytes;
		if (encoding.equals("gzip"))
			return gunzip(bytes);
		throw new N5Exception("unsupported sharding encoding: " + encoding);
	}

	private static byte[] gunzip(final byte[] bytes) throws N5IOException {

		try (final GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
			final ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, bytes.length * 2));
			final byte[] buf = new byte[8192];
			int r;
			while ((r = in.read(buf)) > 0)
				out.write(buf, 0, r);
			return out.toByteArray();
		} catch (final IOException e) {
			throw new N5IOException("failed to gunzip sharded data", e);
		}
	}
}
