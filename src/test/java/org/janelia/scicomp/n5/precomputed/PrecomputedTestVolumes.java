package org.janelia.scicomp.n5.precomputed;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import javax.imageio.ImageIO;

import org.janelia.saalfeldlab.n5.DataType;

/**
 * Helpers to write small precomputed volumes to a directory for tests.
 */
public class PrecomputedTestVolumes {

	@FunctionalInterface
	public interface ValueFn {

		/** value at voxel (x, y, z) for channel c. */
		long apply(long x, long y, long z, int c);
	}

	private PrecomputedTestVolumes() {}

	/**
	 * Writes an {@code info} file with a single {@code raw} scale, then writes
	 * all (clamped) chunk files with data given by {@code fn}, in little-endian
	 * {@code [x,y,z,channel]} Fortran order.
	 */
	public static void writeRawVolume(
			final Path dir,
			final DataType dataType,
			final int numChannels,
			final long[] size,        // [x,y,z]
			final int[] chunk,        // [x,y,z]
			final long[] voxelOffset, // [x,y,z]
			final ValueFn fn) {

		try {
			final String key = "1_1_1";
			Files.createDirectories(dir);
			writeInfo(dir, "image", dataType, numChannels, key, size, chunk, voxelOffset, "raw", null);

			final Path scaleDir = dir.resolve(key);
			Files.createDirectories(scaleDir);

			final int bytes = PrecomputedDataType.bytesPerElement(dataType);
			final long[] grid = new long[3];
			for (int d = 0; d < 3; ++d)
				grid[d] = (size[d] + chunk[d] - 1) / chunk[d];

			for (long gz = 0; gz < grid[2]; ++gz)
				for (long gy = 0; gy < grid[1]; ++gy)
					for (long gx = 0; gx < grid[0]; ++gx) {

						final long[] g = {gx, gy, gz};
						final long[] begin = new long[3];
						final long[] end = new long[3];
						final int[] s = new int[3];
						for (int d = 0; d < 3; ++d) {
							begin[d] = voxelOffset[d] + g[d] * chunk[d];
							end[d] = voxelOffset[d] + Math.min((g[d] + 1) * (long)chunk[d], size[d]);
							s[d] = (int)(end[d] - begin[d]);
						}

						final int n = s[0] * s[1] * s[2] * numChannels;
						final ByteBuffer buf = ByteBuffer.allocate(n * bytes).order(ByteOrder.LITTLE_ENDIAN);
						for (int c = 0; c < numChannels; ++c)
							for (int z = 0; z < s[2]; ++z)
								for (int y = 0; y < s[1]; ++y)
									for (int x = 0; x < s[0]; ++x)
										put(buf, dataType, fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c));

						final String name = begin[0] + "-" + end[0] + "_" + begin[1] + "-" + end[1] + "_" + begin[2] + "-" + end[2];
						Files.write(scaleDir.resolve(name), buf.array());
					}
		} catch (final IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	/**
	 * Writes an {@code info} file with a single image-encoded ({@code png} or
	 * {@code jpeg}) scale, and one image file per chunk. Each chunk image has
	 * width {@code sx} and height {@code sy*sz}, so that a row-major scan equals
	 * the {@code [x,y,z]} Fortran flattening. Supports (uint8,1ch), (uint16,1ch)
	 * and (uint8,3ch).
	 */
	public static void writeImageVolume(
			final Path dir,
			final DataType dataType,
			final int numChannels,
			final long[] size,
			final int[] chunk,
			final long[] voxelOffset,
			final String encoding, // "png" or "jpeg"
			final ValueFn fn) {

		try {
			final String key = "1_1_1";
			Files.createDirectories(dir);
			writeInfo(dir, "image", dataType, numChannels, key, size, chunk, voxelOffset, encoding, null);
			final Path scaleDir = dir.resolve(key);
			Files.createDirectories(scaleDir);

			final String format = encoding.equals("jpeg") ? "jpg" : "png";
			final long[] grid = new long[3];
			for (int d = 0; d < 3; ++d)
				grid[d] = (size[d] + chunk[d] - 1) / chunk[d];

			for (long gz = 0; gz < grid[2]; ++gz)
				for (long gy = 0; gy < grid[1]; ++gy)
					for (long gx = 0; gx < grid[0]; ++gx) {

						final long[] g = {gx, gy, gz};
						final long[] begin = new long[3];
						final long[] end = new long[3];
						final int[] s = new int[3];
						for (int d = 0; d < 3; ++d) {
							begin[d] = voxelOffset[d] + g[d] * chunk[d];
							end[d] = voxelOffset[d] + Math.min((g[d] + 1) * (long)chunk[d], size[d]);
							s[d] = (int)(end[d] - begin[d]);
						}

						final int width = s[0];
						final int height = s[1] * s[2];
						final BufferedImage img = makeImage(width, height, dataType, numChannels);
						for (int z = 0; z < s[2]; ++z)
							for (int y = 0; y < s[1]; ++y)
								for (int x = 0; x < s[0]; ++x) {
									final int xi = x;
									final int yi = y + s[1] * z;
									if (numChannels == 3) {
										final int r = (int)(fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, 0) & 0xff);
										final int gg = (int)(fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, 1) & 0xff);
										final int b = (int)(fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, 2) & 0xff);
										img.setRGB(xi, yi, (r << 16) | (gg << 8) | b);
									} else {
										final int mask = dataType == DataType.UINT16 ? 0xffff : 0xff;
										img.getRaster().setSample(xi, yi, 0, (int)(fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, 0) & mask));
									}
								}

						final String name = begin[0] + "-" + end[0] + "_" + begin[1] + "-" + end[1] + "_" + begin[2] + "-" + end[2];
						ImageIO.write(img, format, scaleDir.resolve(name).toFile());
					}
		} catch (final IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	private static BufferedImage makeImage(final int width, final int height, final DataType dataType, final int numChannels) {

		if (numChannels == 3)
			return new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		if (dataType == DataType.UINT16 || dataType == DataType.INT16)
			return new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
		return new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
	}

	/**
	 * Returns the raw little-endian {@code [x,y,z,channel]} Fortran-order bytes
	 * for a (clamped) chunk of the given size, values from {@code fn}.
	 *
	 * @param s     clamped chunk size {@code [x,y,z]}
	 * @param begin absolute voxel origin {@code [x,y,z]}
	 */
	public static byte[] rawChunkBytes(
			final DataType dataType,
			final int numChannels,
			final int[] s,
			final long[] begin,
			final ValueFn fn) {

		final int bytes = PrecomputedDataType.bytesPerElement(dataType);
		final int n = s[0] * s[1] * s[2] * numChannels;
		final ByteBuffer buf = ByteBuffer.allocate(n * bytes).order(ByteOrder.LITTLE_ENDIAN);
		for (int c = 0; c < numChannels; ++c)
			for (int z = 0; z < s[2]; ++z)
				for (int y = 0; y < s[1]; ++y)
					for (int x = 0; x < s[0]; ++x)
						put(buf, dataType, fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c));
		return buf.array();
	}

	public static void writeInfo(
			final Path dir,
			final String type,
			final DataType dataType,
			final int numChannels,
			final String key,
			final long[] size,
			final int[] chunk,
			final long[] voxelOffset,
			final String encoding,
			final String extraScaleFields) throws IOException {

		final StringBuilder sb = new StringBuilder();
		sb.append("{\n");
		sb.append("  \"@type\": \"neuroglancer_multiscale_volume\",\n");
		sb.append("  \"type\": \"").append(type).append("\",\n");
		sb.append("  \"data_type\": \"").append(dataType.toString()).append("\",\n");
		sb.append("  \"num_channels\": ").append(numChannels).append(",\n");
		sb.append("  \"scales\": [\n");
		sb.append("    {\n");
		sb.append("      \"key\": \"").append(key).append("\",\n");
		sb.append("      \"size\": [").append(size[0]).append(", ").append(size[1]).append(", ").append(size[2]).append("],\n");
		sb.append("      \"resolution\": [1, 1, 1],\n");
		sb.append("      \"voxel_offset\": [").append(voxelOffset[0]).append(", ").append(voxelOffset[1]).append(", ").append(voxelOffset[2]).append("],\n");
		sb.append("      \"chunk_sizes\": [[").append(chunk[0]).append(", ").append(chunk[1]).append(", ").append(chunk[2]).append("]],\n");
		if (extraScaleFields != null)
			sb.append(extraScaleFields);
		sb.append("      \"encoding\": \"").append(encoding).append("\"\n");
		sb.append("    }\n");
		sb.append("  ]\n");
		sb.append("}\n");
		Files.write(dir.resolve("info"), sb.toString().getBytes());
	}

	private static void put(final ByteBuffer buf, final DataType dataType, final long value) {

		switch (dataType) {
		case UINT8:
		case INT8:
			buf.put((byte)value);
			break;
		case UINT16:
		case INT16:
			buf.putShort((short)value);
			break;
		case UINT32:
		case INT32:
			buf.putInt((int)value);
			break;
		case UINT64:
		case INT64:
			buf.putLong(value);
			break;
		case FLOAT32:
			buf.putFloat(value);
			break;
		case FLOAT64:
			buf.putDouble(value);
			break;
		default:
			throw new IllegalArgumentException("unsupported: " + dataType);
		}
	}
}
