package org.janelia.saalfeldlab.n5.precomputed;

import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.RawCompression;
import org.janelia.saalfeldlab.n5.precomputed.PrecomputedInfo.Scale;
import org.janelia.saalfeldlab.n5.precomputed.PrecomputedInfo.Sharding;

/**
 * {@link DatasetAttributes} for a single Neuroglancer precomputed scale.
 * <p>
 * The N5 model is 4-dimensional {@code [x, y, z, channel]}: precomputed raw
 * chunks are little-endian Fortran order {@code [x, y, z, channel]} (x fastest),
 * which matches N5/ImgLib2 column-major exactly, so no axis reversal is needed.
 * The channel axis is never chunked (one block spans all channels).
 * <p>
 * n5&nbsp;3.x has no codec pipeline, so the chunk decoding (encoding, endianness,
 * edge clamping) is done by the reader via
 * {@link org.janelia.saalfeldlab.n5.precomputed.codec.PrecomputedChunkDecoder};
 * the {@link DatasetAttributes} compression is just {@link RawCompression}.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedDatasetAttributes extends DatasetAttributes {

	private static final long serialVersionUID = 1L;

	private final String key;
	private final String encoding;
	private final long[] spatialSize;      // [x, y, z]
	private final int[] spatialChunkSize;  // [x, y, z]
	private final long[] voxelOffset;      // [x, y, z]
	private final int numChannels;
	private final int[] compressedSegmentationBlockSize; // [x, y, z] or null
	private final Sharding sharding;

	public PrecomputedDatasetAttributes(final PrecomputedInfo info, final Scale scale) {

		super(
				dimensions(scale, info.numChannels),
				blockSize(scale, info.numChannels),
				dataType(info),
				new RawCompression());

		this.key = scale.key;
		this.encoding = scale.encoding;
		this.spatialSize = scale.size.clone();
		this.spatialChunkSize = scale.getChunkSize().clone();
		this.voxelOffset = scale.getVoxelOffset().clone();
		this.numChannels = info.numChannels;
		this.compressedSegmentationBlockSize = scale.compressedSegmentationBlockSize;
		this.sharding = scale.sharding;
	}

	private static DataType dataType(final PrecomputedInfo info) {

		final DataType dataType = PrecomputedDataType.fromString(info.dataType);
		if (dataType == null)
			throw new IllegalArgumentException("unsupported precomputed data_type: " + info.dataType);
		return dataType;
	}

	private static long[] dimensions(final Scale scale, final int numChannels) {

		final long[] s = scale.size;
		return new long[]{s[0], s[1], s[2], numChannels};
	}

	private static int[] blockSize(final Scale scale, final int numChannels) {

		final int[] c = scale.getChunkSize();
		return new int[]{c[0], c[1], c[2], numChannels};
	}

	/**
	 * Whether this scale is stored in the sharded format.
	 *
	 * @return true if sharded
	 */
	public boolean isShardedPrecomputed() {

		return sharding != null;
	}

	public Sharding getSharding() {

		return sharding;
	}

	public String getKey() {

		return key;
	}

	public String getEncoding() {

		return encoding;
	}

	public int getNumChannels() {

		return numChannels;
	}

	public int[] getCompressedSegmentationBlockSize() {

		return compressedSegmentationBlockSize;
	}

	/** @return the spatial dataset size {@code [x, y, z]}. */
	public long[] getSpatialSize() {

		return spatialSize;
	}

	/** @return the spatial chunk size {@code [x, y, z]}. */
	public int[] getSpatialChunkSize() {

		return spatialChunkSize;
	}

	/** @return the voxel offset {@code [x, y, z]}. */
	public long[] getVoxelOffset() {

		return voxelOffset;
	}

	/**
	 * Number of chunks along each spatial axis, {@code ceil(size / chunk)}.
	 *
	 * @return the grid size {@code [x, y, z]}
	 */
	public long[] getGridSize() {

		final long[] grid = new long[3];
		for (int d = 0; d < 3; ++d)
			grid[d] = (spatialSize[d] + spatialChunkSize[d] - 1) / spatialChunkSize[d];
		return grid;
	}

	/**
	 * Clamped block size for a grid position {@code [x, y, z, channel]}.
	 * Precomputed stores edge chunks at their true (clamped) extent; the channel
	 * axis is never clamped.
	 */
	public int[] clampedBlockSize(final long... gridPosition) {

		final int[] clamped = new int[4];
		for (int d = 0; d < 3; ++d) {
			final long start = gridPosition[d] * (long)spatialChunkSize[d];
			final long remain = spatialSize[d] - start;
			clamped[d] = (int)Math.min(spatialChunkSize[d], Math.max(0, remain));
		}
		clamped[3] = numChannels;
		return clamped;
	}

	/**
	 * Constructs the unsharded chunk key
	 * {@code xBegin-xEnd_yBegin-yEnd_zBegin-zEnd} for a block grid position.
	 * Only the first three (spatial) grid coordinates are used; the channel
	 * coordinate is always 0.
	 */
	public String chunkKey(final long... gridPosition) {

		final StringBuilder sb = new StringBuilder();
		for (int d = 0; d < 3; ++d) {
			if (d > 0)
				sb.append('_');
			final long begin = voxelOffset[d] + gridPosition[d] * spatialChunkSize[d];
			final long end = voxelOffset[d] + Math.min((gridPosition[d] + 1) * (long)spatialChunkSize[d], spatialSize[d]);
			sb.append(begin).append('-').append(end);
		}
		return sb.toString();
	}
}
