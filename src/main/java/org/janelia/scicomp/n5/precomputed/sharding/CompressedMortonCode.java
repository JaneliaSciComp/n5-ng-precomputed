package org.janelia.scicomp.n5.precomputed.sharding;

/**
 * Computes the Neuroglancer <em>compressed</em> 3D Morton code for a chunk grid
 * position.
 * <p>
 * A compressed Morton code is a Morton (Z-order) code that omits the bits that
 * would always be zero given the grid size along each axis. For each bit-plane
 * {@code i} (from LSB), the bit {@code i} of each axis is appended (x, then y,
 * then z) only if axis has at least {@code i+1} significant bits.
 *
 * @author Stephan Preibisch
 */
public class CompressedMortonCode {

	private CompressedMortonCode() {}

	/**
	 * @param position the chunk grid position (only the first 3 elements are used)
	 * @param gridSize the number of chunks along each axis {@code [x, y, z]}
	 * @return the compressed Morton code
	 */
	public static long encode(final long[] position, final long[] gridSize) {

		final int[] bits = new int[3];
		int maxBits = 0;
		for (int d = 0; d < 3; ++d) {
			bits[d] = gridSize[d] <= 1 ? 0 : (64 - Long.numberOfLeadingZeros(gridSize[d] - 1));
			maxBits = Math.max(maxBits, bits[d]);
		}

		long code = 0;
		int j = 0;
		for (int i = 0; i < maxBits; ++i)
			for (int d = 0; d < 3; ++d)
				if (i < bits[d]) {
					final long bit = (position[d] >>> i) & 1L;
					code |= bit << j;
					++j;
				}
		return code;
	}
}
