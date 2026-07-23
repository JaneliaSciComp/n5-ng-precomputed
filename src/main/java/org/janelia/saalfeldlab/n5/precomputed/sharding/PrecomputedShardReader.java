package org.janelia.saalfeldlab.n5.precomputed.sharding;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.zip.GZIPInputStream;

import org.janelia.saalfeldlab.n5.KeyValueAccess;
import org.janelia.saalfeldlab.n5.LockedChannel;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5Exception.N5IOException;
import org.janelia.saalfeldlab.n5.precomputed.PrecomputedDatasetAttributes;
import org.janelia.saalfeldlab.n5.precomputed.PrecomputedInfo.Sharding;

/**
 * Reads chunks from a Neuroglancer precomputed <em>sharded</em> scale.
 * <p>
 * Given a chunk grid position, this computes the compressed Morton code, maps it
 * to a shard + minishard using the {@link Sharding} parameters, then reads the
 * shard index entry, the minishard index, and finally the chunk bytes from the
 * shard file, returning the (data_encoding-decoded) chunk bytes — or {@code null}
 * if the chunk is absent. The caller decodes those bytes per the scale encoding.
 * <p>
 * n5&nbsp;3.x has no range-read primitive ({@code ReadData.slice}), so byte
 * ranges are read by opening the shard's {@link InputStream} and discarding the
 * leading {@code offset} bytes. This is correct but not bandwidth-optimal on
 * cloud backends (the leading bytes are still transferred); the n5-4.x branch
 * uses true range reads.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedShardReader {

	private PrecomputedShardReader() {}

	/**
	 * @return the chunk's (data_encoding-decoded) bytes, or {@code null} if the
	 *         chunk is not present in the shard
	 */
	public static byte[] readChunkBytes(
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
		return decode(chunkRaw, sh.dataEncoding);
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

		try (final LockedChannel ch = kva.lockForReading(path);
				final InputStream in = ch.newInputStream()) {
			skipFully(in, offset);
			return readFully(in, (int)length);
		} catch (final IOException e) {
			throw new N5IOException("failed to read range from " + path, e);
		}
	}

	private static void skipFully(final InputStream in, final long n) throws IOException {

		final byte[] buf = new byte[8192];
		long remaining = n;
		while (remaining > 0) {
			final int r = in.read(buf, 0, (int)Math.min(buf.length, remaining));
			if (r < 0)
				throw new IOException("unexpected EOF while skipping to offset");
			remaining -= r;
		}
	}

	private static byte[] readFully(final InputStream in, final int length) throws IOException {

		final byte[] out = new byte[length];
		int off = 0;
		while (off < length) {
			final int r = in.read(out, off, length - off);
			if (r < 0)
				throw new IOException("unexpected EOF: wanted " + length + " bytes, got " + off);
			off += r;
		}
		return out;
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
