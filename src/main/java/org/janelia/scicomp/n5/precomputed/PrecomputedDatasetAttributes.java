package org.janelia.scicomp.n5.precomputed;

import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.scicomp.n5.precomputed.PrecomputedInfo.Scale;
import org.janelia.scicomp.n5.precomputed.PrecomputedInfo.Sharding;
import org.janelia.scicomp.n5.precomputed.codec.PrecomputedBlockCodecInfo;

/**
 * {@link DatasetAttributes} for a single Neuroglancer precomputed scale.
 * <p>
 * The N5 model is 4-dimensional {@code [x, y, z, channel]}: precomputed raw
 * chunks are little-endian Fortran order {@code [x, y, z, channel]} (x fastest),
 * which matches N5/ImgLib2 column-major exactly, so no axis reversal is needed.
 * The channel axis is never chunked (one block spans all channels).
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
	private final Sharding sharding;

	public PrecomputedDatasetAttributes(final PrecomputedInfo info, final Scale scale) {

		super(
				dimensions(scale, info.numChannels),
				blockSize(scale, info.numChannels),
				dataType(info),
				new PrecomputedBlockCodecInfo(
						scale.encoding,
						dimensions(scale, info.numChannels),
						info.numChannels,
						scale.compressedSegmentationBlockSize));

		this.key = scale.key;
		this.encoding = scale.encoding;
		this.spatialSize = scale.size.clone();
		this.spatialChunkSize = scale.getChunkSize().clone();
		this.voxelOffset = scale.getVoxelOffset().clone();
		this.numChannels = info.numChannels;
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
	 * Constructs the unsharded chunk key
	 * {@code xBegin-xEnd_yBegin-yEnd_zBegin-zEnd} for a block grid position.
	 * Only the first three (spatial) grid coordinates are used; the channel
	 * coordinate is always 0.
	 */
	@Override
	public String relativeBlockPath(final long... gridPosition) {

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
