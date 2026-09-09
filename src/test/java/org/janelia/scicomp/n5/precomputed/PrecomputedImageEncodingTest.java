package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.scicomp.n5.precomputed.PrecomputedTestVolumes.ValueFn;
import org.junit.Test;

public class PrecomputedImageEncodingTest {

	private static final long[] SIZE = {6, 4, 2};
	private static final int[] CHUNK = {4, 4, 2};
	private static final long[] OFFSET = {0, 0, 0};

	// smooth, channel-independent ramp (good for lossy jpeg)
	private static final ValueFn GRAY = (x, y, z, c) -> 30 + 8 * x + 12 * y + 6 * z;
	// distinct per channel (for lossless channel-order checks)
	private static final ValueFn RGB = (x, y, z, c) -> 30 + 8 * x + 12 * y + 6 * z + 70L * c;
	private static final ValueFn GRAY16 = (x, y, z, c) -> 1000 + 100 * x + 200 * y + 300 * z;

	@Test
	public void testPngUint8Gray() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-png-u8");
		PrecomputedTestVolumes.writeImageVolume(dir, DataType.UINT8, 1, SIZE, CHUNK, OFFSET, "png", GRAY);
		verify(dir, DataType.UINT8, 1, GRAY, 0);
	}

	@Test
	public void testPngUint16Gray() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-png-u16");
		PrecomputedTestVolumes.writeImageVolume(dir, DataType.UINT16, 1, SIZE, CHUNK, OFFSET, "png", GRAY16);
		verify(dir, DataType.UINT16, 1, GRAY16, 0);
	}

	@Test
	public void testPngRgb() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-png-rgb");
		PrecomputedTestVolumes.writeImageVolume(dir, DataType.UINT8, 3, SIZE, CHUNK, OFFSET, "png", RGB);
		verify(dir, DataType.UINT8, 3, RGB, 0);
	}

	@Test
	public void testJpegGray() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-jpeg-u8");
		PrecomputedTestVolumes.writeImageVolume(dir, DataType.UINT8, 1, SIZE, CHUNK, OFFSET, "jpeg", GRAY);
		verify(dir, DataType.UINT8, 1, GRAY, 15);
	}

	@Test
	public void testJpegRgb() throws IOException {

		final Path dir = Files.createTempDirectory("n5-pc-jpeg-rgb");
		PrecomputedTestVolumes.writeImageVolume(dir, DataType.UINT8, 3, SIZE, CHUNK, OFFSET, "jpeg", GRAY);
		verify(dir, DataType.UINT8, 3, GRAY, 15);
	}

	private static void verify(
			final Path dir,
			final DataType dataType,
			final int numChannels,
			final ValueFn fn,
			final long tolerance) {

		final N5PrecomputedReader n5 = new N5PrecomputedReader(dir.toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertEquals(dataType, attrs.getDataType());
		assertArrayEquals(new long[]{SIZE[0], SIZE[1], SIZE[2], numChannels}, attrs.getDimensions());

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (SIZE[d] + CHUNK[d] - 1) / CHUNK[d];

		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {

					final DataBlock<?> block = n5.readBlock("1_1_1", attrs, gx, gy, gz, 0);
					final long[] g = {gx, gy, gz};
					final long[] begin = new long[3];
					final int[] s = new int[3];
					for (int d = 0; d < 3; ++d) {
						begin[d] = g[d] * CHUNK[d];
						s[d] = (int)Math.min(CHUNK[d], SIZE[d] - begin[d]);
					}
					assertArrayEquals(new int[]{s[0], s[1], s[2], numChannels}, block.getSize());

					int idx = 0;
					for (int c = 0; c < numChannels; ++c)
						for (int z = 0; z < s[2]; ++z)
							for (int y = 0; y < s[1]; ++y)
								for (int x = 0; x < s[0]; ++x, ++idx) {
									final long expected = fn.apply(begin[0] + x, begin[1] + y, begin[2] + z, c) & 0xffff;
									final long actual = elementAt(block, idx);
									assertTrue("value @ block " + gx + "," + gy + "," + gz + " idx " + idx
													+ " expected~" + expected + " actual " + actual,
											Math.abs(actual - expected) <= tolerance);
								}
				}
	}

	private static long elementAt(final DataBlock<?> block, final int idx) {

		final Object data = block.getData();
		if (data instanceof byte[])
			return ((byte[])data)[idx] & 0xffL;
		if (data instanceof short[])
			return ((short[])data)[idx] & 0xffffL;
		throw new IllegalArgumentException("unexpected data type");
	}
}
