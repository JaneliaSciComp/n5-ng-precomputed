package org.janelia.scicomp.n5.precomputed;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.view.Views;

/**
 * Real-life smoke test against the public Google Cloud Storage hemibrain
 * precomputed volume (JPEG + sharded). Opt-in and network-dependent, so it is
 * skipped unless run with {@code -Dprecomputed.gcs=true}.
 */
public class PrecomputedGoogleCloudTest {

	@BeforeClass
	public static void requireOptIn() {

		Assume.assumeTrue("set -Dprecomputed.gcs=true to run the GCS smoke test",
				Boolean.getBoolean("precomputed.gcs"));
	}

	@Test
	public void testOpenAndReadFinestAndCoarsest() {

		final N5Reader n5 = HemibrainPrecomputed.open(HemibrainPrecomputed.URL);

		final String[] keys = n5.list("/");
		assertTrue("expected several scale levels", keys.length >= 2);

		// finest scale metadata
		final DatasetAttributes s0 = n5.getDatasetAttributes(keys[0]);
		assertEquals(DataType.UINT8, s0.getDataType());
		assertEquals("expected 4D [x,y,z,channel]", 4, s0.getNumDimensions());
		assertEquals("single channel", 1, s0.getDimensions()[3]);
		assertArrayEquals(new int[]{64, 64, 64, 1}, s0.getBlockSize());
		assertTrue(((PrecomputedDatasetAttributes)s0).isShardedPrecomputed());
		assertEquals("jpeg", ((PrecomputedDatasetAttributes)s0).getEncoding());

		// read one JPEG chunk from the coarsest (smallest) scale over the sharded backend
		final String coarsest = keys[keys.length - 1];
		final DatasetAttributes sN = n5.getDatasetAttributes(coarsest);
		final DataBlock<?> block = n5.readBlock(coarsest, sN, 0, 0, 0, 0);
		assertNotNull("chunk (0,0,0) of coarsest scale should exist", block);
		assertTrue("uint8 chunk decodes to byte[]", block.getData() instanceof byte[]);
		final int[] bs = block.getSize();
		assertEquals(((byte[])block.getData()).length, bs[0] * bs[1] * bs[2] * bs[3]);

		// and via imglib2 (exercises N5Utils volatile cache + hyperSlice)
		final RandomAccessibleInterval<UnsignedByteType> img =
				Views.hyperSlice(N5UtilsOpen(n5, coarsest), 3, 0);
		assertEquals(3, img.numDimensions());
		final RandomAccess<UnsignedByteType> ra = img.randomAccess();
		ra.setPosition(new long[]{0, 0, 0});
		final int v = ra.get().get();
		assertTrue(v >= 0 && v <= 255);
	}

	private static RandomAccessibleInterval N5UtilsOpen(final N5Reader n5, final String key) {

		return N5Utils.openVolatile(n5, key);
	}
}
