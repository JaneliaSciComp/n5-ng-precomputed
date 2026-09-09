package org.janelia.scicomp.n5.precomputed;

import java.awt.GraphicsEnvironment;
import java.util.Arrays;
import java.util.concurrent.ForkJoinPool;

import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.imglib2.N5Utils;

import bdv.cache.SharedQueue;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.util.volatiles.VolatileViews;
import ij.ImageJ;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.type.volatiles.VolatileFloatType;
import net.imglib2.view.Views;

/**
 * Real-life example: open a Neuroglancer precomputed <em>deformation / warp
 * field</em> from Google Cloud Storage and display its center z-slice in
 * BigDataViewer as per-channel 2D volatile sources.
 * <p>
 * The volume at {@code gs://janelia-spark-test/warp_flow_precomputed/w61_s109_r00}
 * is {@code float32}, 2 channels (the x/y displacement components), single scale,
 * raw-encoded and sharded with {@code data_encoding: gzip} — so this exercises the
 * float32 + sharded + gzip-data read path over a cloud backend. The reader presents
 * it as a 4D {@code [x, y, z, channel]} dataset; the center z-slice is hyper-sliced
 * out and each channel is shown as its own lazily-rendered 2D volatile BDV source
 * ({@code is2D()}).
 * <p>
 * Run interactively:
 * <pre>
 * mvn test-compile exec:java -Dexec.classpathScope=test \
 *   -Dexec.mainClass=org.janelia.scicomp.n5.precomputed.WarpFieldPrecomputed
 * </pre>
 *
 * @author Stephan Preibisch
 */
public class WarpFieldPrecomputed {

	public static final String URL =
			"precomputed://gs://janelia-spark-test/warp_flow_precomputed/w61_s109_r00";

	public static void main(final String[] args) {

		final String url = args.length > 0 ? args[0] : URL;
		final N5Reader n5 = HemibrainPrecomputed.open(url);

		final String key = n5.list("/")[0];
		final DatasetAttributes attrs = n5.getDatasetAttributes(key);
		System.out.println("scale " + key + ": " + attrs.getDataType()
				+ " " + Arrays.toString(attrs.getDimensions())
				+ " block " + Arrays.toString(attrs.getBlockSize()));

		final double[] resolution = n5.getAttribute(key, "resolution", double[].class);

		// volatile-cached 4D [x, y, z, channel] view of the field
		final RandomAccessibleInterval<FloatType> field = N5Utils.openVolatile(n5, key);
		final long[] dims = field.dimensionsAsLongArray();
		final int numChannels = (int)dims[3];
		final long centerZ = 42; // center section of the 82-slice stack

		// probe the center voxel to force a real (sharded, gzip, float32) chunk read
		final RandomAccess<FloatType> ra = field.randomAccess();
		ra.setPosition(new long[]{dims[0] / 2, dims[1] / 2, centerZ, 0});
		System.out.println("sample @ (center, z=" + centerZ + ", ch=0) = " + ra.get().get());

		if (GraphicsEnvironment.isHeadless()) {
			System.out.println("headless environment: skipping display");
			return;
		}

		boolean imageJ = false;

		// ImageJ is impractical for the full volume: ImageJFunctions.show() eagerly
		// pulls every plane into an ImagePlus, so even with a multi-threaded
		// ForkJoinPool it loads the whole (sharded, gzip, float32) volume up front and
		// is slow. 
		if ( imageJ )
		{
			new ImageJ();
			ImageJFunctions.show(field, new ForkJoinPool(32));
		}
		else {
			// Pick the center z-slice -> a 2-channel 2D image, and show each channel as its
			// own lazily-rendered volatile source in a 2D BDV. is2D() keeps the viewer
			// axis-aligned (2D transform handler, no out-of-plane rotation) so the single
			// slice can't be lost in 3D. Calibrated with the x/y resolution.
			final SharedQueue queue = new SharedQueue(Math.max(1, Runtime.getRuntime().availableProcessors() - 1));
			final RandomAccessibleInterval<VolatileFloatType> volatileField = VolatileViews.wrapAsVolatile(field, queue);
			final RandomAccessibleInterval<VolatileFloatType> slice = Views.hyperSlice(volatileField, 2, centerZ);
	
			final AffineTransform3D sourceTransform = new AffineTransform3D();
			sourceTransform.set(
					resolution[0], 0, 0, 0,
					0, resolution[1], 0, 0,
					0, 0, 1, 0);
	
			BdvStackSource<?> bdv = null;
			for (int c = 0; c < numChannels; ++c) {
				final RandomAccessibleInterval<VolatileFloatType> channel = Views.hyperSlice(slice, 2, c);
				final BdvOptions options = (bdv == null)
						? BdvOptions.options().is2D().sourceTransform(sourceTransform)
						: BdvOptions.options().addTo(bdv).sourceTransform(sourceTransform);
				bdv = BdvFunctions.show(channel, "w61_s109_r00 z" + centerZ + " ch" + c, options);
				// rough display range for a displacement field; adjust in BDV as needed
				bdv.setDisplayRange(-5, 5);
			}
		}
	}
}
