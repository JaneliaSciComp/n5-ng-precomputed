package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.scicomp.n5.precomputed.PrecomputedTestVolumes.ValueFn;
import org.junit.Test;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedShortType;

public class PrecomputedRawTest {

	private static final ValueFn FN = (x, y, z, c) -> x + 7 * y + 53 * z + 1009L * c;

	@Test
	public void testMetadataAndBlocks() throws IOException {

		final Path dir = Files.createTempDirectory("n5-ng-precomputed-raw");
		final long[] size = {5, 4, 3};
		final int[] chunk = {2, 2, 2};
		final long[] offset = {0, 0, 0};
		final int numChannels = 2;

		PrecomputedTestVolumes.writeRawVolume(dir, DataType.UINT16, numChannels, size, chunk, offset, FN);

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());

		// metadata surface
		assertTrue(n5.exists("/"));
		assertTrue(n5.groupExists("/"));
		assertFalse(n5.datasetExists("/"));
		assertTrue(n5.exists("1_1_1"));
		assertTrue(n5.datasetExists("1_1_1"));
		assertFalse(n5.datasetExists("nope"));
		assertArrayEquals(new String[]{"1_1_1"}, n5.list("/"));

		// attributes
		assertEquals("raw", n5.getAttribute("1_1_1", "encoding", String.class));
		assertEquals(Integer.valueOf(2), n5.getAttribute("", "num_channels", Integer.class));
		assertArrayEquals(size, n5.getAttribute("1_1_1", "size", long[].class));

		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertArrayEquals(new long[]{5, 4, 3, 2}, attrs.getDimensions());
		assertArrayEquals(new int[]{2, 2, 2, 2}, attrs.getBlockSize());
		assertEquals(DataType.UINT16, attrs.getDataType());

		verifyAllBlocks(n5, attrs, size, chunk, offset, numChannels, FN);
		verifyImgLib2(n5, size, offset, numChannels, FN);
	}

	@Test
	public void testVoxelOffset() throws IOException {

		final Path dir = Files.createTempDirectory("n5-ng-precomputed-raw-offset");
		final long[] size = {3, 3, 2};
		final int[] chunk = {2, 2, 2};
		final long[] offset = {10, 20, 30};

		PrecomputedTestVolumes.writeRawVolume(dir, DataType.UINT8, 1, size, chunk, offset, FN);

		// chunk file names must include the voxel offset
		assertTrue(Files.isRegularFile(dir.resolve("1_1_1").resolve("10-12_20-22_30-32")));
		assertTrue(Files.isRegularFile(dir.resolve("1_1_1").resolve("12-13_20-22_30-32")));

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		verifyAllBlocks(n5, attrs, size, chunk, offset, 1, FN);
	}

	@Test
	public void testMissingBlockReturnsNull() throws IOException {

		final Path dir = Files.createTempDirectory("n5-ng-precomputed-raw-missing");
		final long[] size = {4, 2, 2};
		final int[] chunk = {2, 2, 2};
		final long[] offset = {0, 0, 0};

		PrecomputedTestVolumes.writeRawVolume(dir, DataType.UINT8, 1, size, chunk, offset, FN);
		Files.delete(dir.resolve("1_1_1").resolve("0-2_0-2_0-2"));

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertNull(n5.readBlock("1_1_1", attrs, 0, 0, 0, 0));
		// the sibling block still exists
		assertTrue(n5.readBlock("1_1_1", attrs, 1, 0, 0, 0) != null);
	}

	private static void verifyAllBlocks(
			final N5PrecomputedReader n5,
			final DatasetAttributes attrs,
			final long[] size,
			final int[] chunk,
			final long[] offset,
			final int numChannels,
			final ValueFn fn) {

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (size[d] + chunk[d] - 1) / chunk[d];

		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {

					final DataBlock<?> block = n5.readBlock("1_1_1", attrs, gx, gy, gz, 0);
					final long[] g = {gx, gy, gz};
					final int[] s = new int[3];
					final long[] begin = new long[3];
					for (int d = 0; d < 3; ++d) {
						begin[d] = offset[d] + g[d] * chunk[d];
						final long end = offset[d] + Math.min((g[d] + 1) * (long)chunk[d], size[d]);
						s[d] = (int)(end - begin[d]);
					}
					assertArrayEquals("block size @ " + gx + "," + gy + "," + gz,
							new int[]{s[0], s[1], s[2], numChannels}, block.getSize());

					int idx = 0;
					for (int c = 0; c < numChannels; ++c)
						for (int z = 0; z < s[2]; ++z)
							for (int y = 0; y < s[1]; ++y)
								for (int x = 0; x < s[0]; ++x, ++idx) {
									final long expected = mask(fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c), attrs.getDataType());
									assertEquals("value @ block " + gx + "," + gy + "," + gz + " idx " + idx,
											expected, elementAt(block, idx));
								}
				}
	}

	private static long mask(final long v, final DataType dataType) {

		switch (dataType) {
		case UINT8:
		case INT8:
			return v & 0xffL;
		case UINT16:
		case INT16:
			return v & 0xffffL;
		case UINT32:
		case INT32:
			return v & 0xffffffffL;
		default:
			return v;
		}
	}

	private static long elementAt(final DataBlock<?> block, final int idx) {

		final Object data = block.getData();
		if (data instanceof byte[])
			return ((byte[])data)[idx] & 0xffL;
		if (data instanceof short[])
			return ((short[])data)[idx] & 0xffffL;
		if (data instanceof int[])
			return ((int[])data)[idx] & 0xffffffffL;
		if (data instanceof long[])
			return ((long[])data)[idx];
		throw new IllegalArgumentException("unexpected data type");
	}

	private static void verifyImgLib2(
			final N5PrecomputedReader n5,
			final long[] size,
			final long[] offset,
			final int numChannels,
			final ValueFn fn) {

		final RandomAccessibleInterval<UnsignedShortType> img = N5Utils.open(n5, "1_1_1");
		assertArrayEquals(new long[]{size[0], size[1], size[2], numChannels}, img.dimensionsAsLongArray());

		final RandomAccess<UnsignedShortType> ra = img.randomAccess();
		for (int c = 0; c < numChannels; ++c)
			for (long z = 0; z < size[2]; ++z)
				for (long y = 0; y < size[1]; ++y)
					for (long x = 0; x < size[0]; ++x) {
						ra.setPosition(new long[]{x, y, z, c});
						final long expected = fn.apply(offset[0] + x, offset[1] + y, offset[2] + z, c);
						assertEquals("imglib2 @ " + x + "," + y + "," + z + "," + c, expected, ra.get().get());
					}
	}
}
