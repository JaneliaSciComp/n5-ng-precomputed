package org.janelia.scicomp.n5.precomputed;

import java.util.ArrayList;
import java.util.List;

import com.google.gson.annotations.SerializedName;

/**
 * In-memory representation of a Neuroglancer precomputed {@code info} JSON
 * document for a volume.
 * <p>
 * See <a href=
 * "https://neuroglancer-docs.web.app/datasource/precomputed/volume.html">the
 * precomputed volume format specification</a>. All spatial fields
 * ({@code size}, {@code resolution}, {@code voxel_offset}, {@code chunk_sizes})
 * are in {@code [x, y, z]} order.
 * <p>
 * Deserialized directly by Gson via matching field names /
 * {@link SerializedName} annotations.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedInfo {

	@SerializedName("@type")
	public String atType;

	/** {@code "image"} or {@code "segmentation"}. */
	public String type;

	/** {@code uint8, int8, uint16, int16, uint32, int32, uint64, float32}. */
	@SerializedName("data_type")
	public String dataType;

	@SerializedName("num_channels")
	public int numChannels = 1;

	public List<Scale> scales = new ArrayList<>();

	/**
	 * Returns the scale whose {@code key} equals the given key, or {@code null}
	 * if there is no such scale.
	 *
	 * @param key the scale key
	 * @return the scale or {@code null}
	 */
	public Scale getScale(final String key) {

		if (scales == null)
			return null;
		for (final Scale s : scales)
			if (s.key != null && s.key.equals(key))
				return s;
		return null;
	}

	/**
	 * Returns the list of scale keys, in the order they appear in the
	 * {@code info}.
	 *
	 * @return the scale keys
	 */
	public String[] getScaleKeys() {

		if (scales == null)
			return new String[0];
		final String[] keys = new String[scales.size()];
		for (int i = 0; i < keys.length; ++i)
			keys[i] = scales.get(i).key;
		return keys;
	}

	/**
	 * Metadata for a single resolution level.
	 */
	public static class Scale {

		public String key;

		/** volume dimensions in voxels, {@code [x, y, z]}. */
		public long[] size;

		/** voxel size, {@code [x, y, z]}. */
		public double[] resolution;

		/** volume origin in voxels, {@code [x, y, z]}, default {@code [0,0,0]}. */
		@SerializedName("voxel_offset")
		public long[] voxelOffset;

		/** allowed chunk sizes, each {@code [x, y, z]}. */
		@SerializedName("chunk_sizes")
		public int[][] chunkSizes;

		/** {@code raw, jpeg, compressed_segmentation, png}. */
		public String encoding;

		@SerializedName("compressed_segmentation_block_size")
		public int[] compressedSegmentationBlockSize;

		public Sharding sharding;

		/**
		 * Returns the (first) chunk size for this scale, in {@code [x, y, z]}
		 * order.
		 *
		 * @return the chunk size
		 */
		public int[] getChunkSize() {

			if (chunkSizes == null || chunkSizes.length == 0)
				throw new IllegalStateException("scale '" + key + "' has no chunk_sizes");
			return chunkSizes[0];
		}

		/**
		 * Returns the voxel offset, defaulting to {@code [0,0,0]} if not set.
		 *
		 * @return the voxel offset
		 */
		public long[] getVoxelOffset() {

			return voxelOffset != null ? voxelOffset : new long[]{0, 0, 0};
		}

		public boolean isSharded() {

			return sharding != null;
		}
	}

	/**
	 * Sharding parameters ({@code neuroglancer_uint64_sharded_v1}).
	 */
	public static class Sharding {

		@SerializedName("@type")
		public String atType;

		@SerializedName("preshift_bits")
		public int preshiftBits;

		/** {@code "identity"} or {@code "murmurhash3_x86_128"}. */
		public String hash = "identity";

		@SerializedName("minishard_bits")
		public int minishardBits;

		@SerializedName("shard_bits")
		public int shardBits;

		/** {@code "raw"} or {@code "gzip"}. */
		@SerializedName("minishard_index_encoding")
		public String minishardIndexEncoding = "raw";

		/** {@code "raw"} or {@code "gzip"}. */
		@SerializedName("data_encoding")
		public String dataEncoding = "raw";
	}
}
