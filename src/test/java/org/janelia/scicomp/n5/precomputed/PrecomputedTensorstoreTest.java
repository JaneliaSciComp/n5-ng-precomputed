package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Cross-validation against volumes written by Google's tensorstore
 * (independent {@code neuroglancer_precomputed} writer). Volumes are filled
 * with deterministic formulas shared with {@code src/test/python/gen_precomputed.py};
 * this test reads them with {@link N5PrecomputedReader} and checks every voxel.
 * <p>
 * Point it at a pre-generated directory with {@code -Dprecomputed.dir=/path};
 * otherwise the test is skipped.
 */
public class PrecomputedTensorstoreTest {

	private static Path root;
	private static JsonArray manifest;

	@BeforeClass
	public static void setup() throws IOException, InterruptedException {

		String dir = System.getProperty("precomputed.dir");
		if (dir == null) {
			// optionally generate the data via python (needs tensorstore + numpy)
			final String python = System.getProperty("precomputed.python");
			Assume.assumeTrue("set -Dprecomputed.dir (pre-generated) or -Dprecomputed.python to run this test", python != null);
			final Path tmp = Files.createTempDirectory("n5-ng-precomputed-ts");
			final Process p = new ProcessBuilder(python, "src/test/python/gen_precomputed.py", tmp.toString())
					.redirectErrorStream(true).inheritIO().start();
			final boolean ok = p.waitFor(180, TimeUnit.SECONDS) && p.exitValue() == 0;
			Assume.assumeTrue("gen_precomputed.py failed (tensorstore/numpy not available?)", ok);
			dir = tmp.toString();
		}
		root = Paths.get(dir);
		final Path manifestPath = root.resolve("manifest.json");
		Assume.assumeTrue("no manifest.json in " + dir, Files.isRegularFile(manifestPath));
		manifest = JsonParser.parseString(new String(Files.readAllBytes(manifestPath))).getAsJsonArray();
	}

	@Test
	public void testAllVolumes() {

		for (int i = 0; i < manifest.size(); ++i) {
			final JsonObject v = manifest.get(i).getAsJsonObject();
			verifyVolume(v);
		}
	}

	private void verifyVolume(final JsonObject v) {

		final String name = v.get("name").getAsString();
		final DataType dataType = DataType.fromString(v.get("dtype").getAsString());
		final String formula = v.get("formula").getAsString();
		final int numChannels = v.get("num_channels").getAsInt();
		final long[] size = longArray(v.getAsJsonArray("size"));
		final int[] chunk = intArray(v.getAsJsonArray("chunk"));

		final N5PrecomputedReader n5 = new N5PrecomputedReader(root.resolve(name).toString());
		final DatasetAttributes attrs = n5.getDatasetAttributes("1_1_1");
		assertEquals(name + " dtype", dataType, attrs.getDataType());
		assertArrayEquals(name + " dims",
				new long[]{size[0], size[1], size[2], numChannels}, attrs.getDimensions());

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (size[d] + chunk[d] - 1) / chunk[d];

		for (long gz = 0; gz < grid[2]; ++gz)
			for (long gy = 0; gy < grid[1]; ++gy)
				for (long gx = 0; gx < grid[0]; ++gx) {
					final DataBlock<?> block = n5.readBlock("1_1_1", attrs, gx, gy, gz, 0);
					final long[] begin = {gx * chunk[0], gy * chunk[1], gz * chunk[2]};
					final int[] s = new int[3];
					for (int d = 0; d < 3; ++d)
						s[d] = (int)Math.min(chunk[d], size[d] - begin[d]);

					int idx = 0;
					for (int c = 0; c < numChannels; ++c)
						for (int z = 0; z < s[2]; ++z)
							for (int y = 0; y < s[1]; ++y)
								for (int x = 0; x < s[0]; ++x, ++idx) {
									final long expected = mask(
											formula(formula, begin[0] + x, begin[1] + y, begin[2] + z, c), dataType);
									final long actual = elementAt(block, idx);
									assertEquals(name + " @ " + gx + "," + gy + "," + gz + " idx " + idx,
											expected, actual);
								}
				}
	}

	// formulas mirrored from gen_precomputed.py
	private static long formula(final String name, final long x, final long y, final long z, final long c) {

		switch (name) {
		case "f_raw":
			return Math.floorMod(x + 7 * y + 53 * z + 1009 * c, 65521);
		case "f_seg":
			return Math.floorMod((x / 4) * 13 + (y / 4) * 7 + (z / 2) * 3 + c, 50);
		case "f_seg64":
			return Math.floorMod((x / 4) + (y / 4) + (z / 2), 7) + (1L << 40);
		default:
			throw new IllegalArgumentException("unknown formula " + name);
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

	private static long[] longArray(final JsonArray a) {

		final long[] r = new long[a.size()];
		for (int i = 0; i < r.length; ++i)
			r[i] = a.get(i).getAsLong();
		return r;
	}

	private static int[] intArray(final JsonArray a) {

		final int[] r = new int[a.size()];
		for (int i = 0; i < r.length; ++i)
			r[i] = a.get(i).getAsInt();
		return r;
	}
}
