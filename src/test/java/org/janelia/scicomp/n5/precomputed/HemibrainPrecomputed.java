package org.janelia.scicomp.n5.precomputed;

import java.net.URI;
import java.util.Arrays;

import org.janelia.saalfeldlab.n5.KeyValueAccess;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.googlecloud.GoogleCloudStorageKeyValueAccess;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;
import org.janelia.saalfeldlab.n5.universe.N5Factory;

import com.google.cloud.storage.Storage;
import com.google.cloud.storage.StorageOptions;
import com.google.gson.GsonBuilder;

import bdv.cache.SharedQueue;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.util.RandomAccessibleIntervalMipmapSource;
import mpicbg.spim.data.sequence.FinalVoxelDimensions;
import mpicbg.spim.data.sequence.VoxelDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.volatiles.VolatileUnsignedByteType;
import net.imglib2.view.Views;

/**
 * Real-life example: open the multi-resolution Neuroglancer precomputed
 * hemibrain EM volume from Google Cloud Storage and display it in
 * BigDataViewer.
 * <p>
 * The volume is JPEG-encoded and sharded, so this exercises the full read path
 * (compressed-Morton shard lookup, gzip minishard index, JPEG chunk decode)
 * over a cloud backend. The multi-resolution source is built in the same style
 * as {@code hot-knife}'s {@code VNCMovie.createMipmapSource}, but the scale
 * levels and downsampling factors are taken from the precomputed {@code info}
 * (scale keys + per-scale {@code resolution}) rather than assumed.
 * <p>
 * Run interactively:
 * <pre>
 * mvn test-compile exec:java -Dexec.classpathScope=test \
 *   -Dexec.mainClass=org.janelia.scicomp.n5.precomputed.HemibrainPrecomputed
 * </pre>
 *
 * @author Stephan Preibisch
 */
public class HemibrainPrecomputed {

	public static final String URL =
			"precomputed://gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg";

	/**
	 * Opens a precomputed volume from a {@code precomputed://<kvstore-url>} (or
	 * plain) URL. {@code gs://} is read anonymously (public buckets); other
	 * schemes go through {@link N5Factory}.
	 * <p>
	 * <b>Future:</b> once {@code n5-universe} knows the precomputed format —
	 * i.e. a {@code StorageFormat.PRECOMPUTED} (prefix {@code precomputed},
	 * detected by the presence of an {@code info} file) plus a matching case in
	 * {@code N5Factory.openReader} that constructs a
	 * {@link PrecomputedKeyValueReader} — this whole method collapses to:
	 * <pre>
	 * return new N5Factory().openReader(url); // url = "precomputed://gs://..."
	 * </pre>
	 * Until then we wire the {@link KeyValueAccess} and reader up by hand (and
	 * open {@code gs://} anonymously, which {@code N5Factory} does not do by
	 * default).
	 *
	 * @param url the (optionally {@code precomputed://}-prefixed) container URL
	 * @return the reader as an {@link N5Reader}
	 */
	public static N5Reader open(final String url) {

		String u = url;
		if (u.startsWith("precomputed://"))
			u = u.substring("precomputed://".length());

		final KeyValueAccess kva;
		if (u.startsWith("gs://")) {
			final Storage storage = StorageOptions.getUnauthenticatedInstance().getService();
			kva = new GoogleCloudStorageKeyValueAccess(storage, u, false);
		} else {
			kva = new N5Factory().getKeyValueAccess(URI.create(u), false);
		}
		return new PrecomputedKeyValueReader(kva, u, new GsonBuilder(), true);
	}

	/**
	 * Builds a BigDataViewer multi-resolution source from all scales of a
	 * (single-channel, uint8) precomputed volume. The trailing singleton channel
	 * axis is dropped.
	 * <p>
	 * Each level gets an explicit {@link AffineTransform3D} (into finest-voxel
	 * world units) that combines three things:
	 * <ol>
	 * <li>the per-axis downsampling factor {@code f = resolution[i]/resolution[0]};</li>
	 * <li>the {@code 0.5*(f-1)} half-pixel correction, so a coarse voxel's center
	 *     coincides with the center of the fine block it summarizes (Neuroglancer
	 *     produces origin-aligned block downsamplings);</li>
	 * <li>the scale's {@code voxel_offset} (times {@code f}), so scales whose
	 *     origin is not a multiple of the downsampling factor still register.</li>
	 * </ol>
	 * For {@code voxel_offset == [0,0,0]} (the common case, e.g. hemibrain) this
	 * reduces to the plain scale + half-pixel transform.
	 */
	@SuppressWarnings({"unchecked", "rawtypes"})
	public static RandomAccessibleIntervalMipmapSource<UnsignedByteType> createMipmapSource(
			final N5Reader n5,
			final String name) {

		final String[] keys = n5.list("/");
		final int numScales = keys.length;

		final RandomAccessibleInterval<UnsignedByteType>[] mipmaps = new RandomAccessibleInterval[numScales];
		final AffineTransform3D[] mipmapTransforms = new AffineTransform3D[numScales];

		final double[] res0 = n5.getAttribute(keys[0], "resolution", double[].class);

		for (int i = 0; i < numScales; ++i) {
			final RandomAccessibleInterval img4d = N5Utils.openVolatile(n5, keys[i]);
			// drop the trailing channel axis (num_channels == 1)
			mipmaps[i] = Views.hyperSlice(img4d, 3, 0);

			final double[] res = n5.getAttribute(keys[i], "resolution", double[].class);
			final long[] off = n5.getAttribute(keys[i], "voxel_offset", long[].class);

			final double fx = res[0] / res0[0], fy = res[1] / res0[1], fz = res[2] / res0[2];
			final double ox = off == null ? 0 : off[0];
			final double oy = off == null ? 0 : off[1];
			final double oz = off == null ? 0 : off[2];

			// downsampling scale + half-pixel correction + voxel_offset origin
			final AffineTransform3D t = new AffineTransform3D();
			t.set(
					fx, 0,  0,  0.5 * (fx - 1) + ox * fx,
					0,  fy, 0,  0.5 * (fy - 1) + oy * fy,
					0,  0,  fz, 0.5 * (fz - 1) + oz * fz);
			mipmapTransforms[i] = t;

			System.out.println(keys[i] + " scale=[" + fx + ", " + fy + ", " + fz + "] voxel_offset="
					+ (off == null ? "[0, 0, 0]" : Arrays.toString(off)));
		}

		final VoxelDimensions voxelDimensions = new FinalVoxelDimensions("nm", res0[0], res0[1], res0[2]);
		return new RandomAccessibleIntervalMipmapSource<>(
				mipmaps, new UnsignedByteType(), mipmapTransforms, voxelDimensions, name, true);
	}

	public static void main(final String[] args) {

		final String url = args.length > 0 ? args[0] : URL;
		final N5Reader n5 = open(url);
		System.out.println("scales: " + Arrays.toString(n5.list("/")));

		final RandomAccessibleIntervalMipmapSource<UnsignedByteType> source = createMipmapSource(n5, "hemibrain / n5-ng-precomputed");

		final SharedQueue queue = new SharedQueue(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
		// pin the volatile type so V is inferrable (the no-type asVolatile(queue)
		// overload leaves V free and breaks show(...) type inference)
		final BdvStackSource<?> bdv = BdvFunctions.show( source.asVolatile( new VolatileUnsignedByteType(), queue ) );

		//final BdvStackSource<?> bdv = BdvFunctions.show(source, BdvOptions.options().numRenderingThreads(Math.max(1, Runtime.getRuntime().availableProcessors() - 1)));

		bdv.setDisplayRange(0, 255);
	}
}
