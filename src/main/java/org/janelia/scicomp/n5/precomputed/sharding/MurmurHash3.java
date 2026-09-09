package org.janelia.scicomp.n5.precomputed.sharding;

/**
 * MurmurHash3 x86_128 (seed 0), as used by the Neuroglancer sharded format for
 * the {@code "murmurhash3_x86_128"} hash. Only the low 64 bits of the 128-bit
 * result are needed.
 *
 * @author Stephan Preibisch
 */
public class MurmurHash3 {

	private static final int C1 = 0x239b961b;
	private static final int C2 = 0xab0e9789;
	private static final int C3 = 0x38b34ae5;
	private static final int C4 = 0xa1e38b93;

	private MurmurHash3() {}

	/**
	 * Hashes a uint64 (encoded as 8 little-endian bytes) with MurmurHash3
	 * x86_128 (seed 0) and returns the low 64 bits of the 128-bit result.
	 *
	 * @param value the value to hash
	 * @return the low 64 bits of the hash
	 */
	public static long hashX86_128Low64(final long value) {

		final byte[] data = new byte[8];
		for (int i = 0; i < 8; ++i)
			data[i] = (byte)(value >>> (8 * i));
		return hashX86_128Low64(data);
	}

	public static long hashX86_128Low64(final byte[] data) {

		int h1 = 0, h2 = 0, h3 = 0, h4 = 0;
		final int len = data.length;
		final int nblocks = len / 16;

		for (int i = 0; i < nblocks; ++i) {
			final int base = i * 16;
			int k1 = le32(data, base);
			int k2 = le32(data, base + 4);
			int k3 = le32(data, base + 8);
			int k4 = le32(data, base + 12);

			k1 *= C1; k1 = Integer.rotateLeft(k1, 15); k1 *= C2; h1 ^= k1;
			h1 = Integer.rotateLeft(h1, 19); h1 += h2; h1 = h1 * 5 + 0x561ccd1b;
			k2 *= C2; k2 = Integer.rotateLeft(k2, 16); k2 *= C3; h2 ^= k2;
			h2 = Integer.rotateLeft(h2, 17); h2 += h3; h2 = h2 * 5 + 0x0bcaa747;
			k3 *= C3; k3 = Integer.rotateLeft(k3, 17); k3 *= C4; h3 ^= k3;
			h3 = Integer.rotateLeft(h3, 15); h3 += h4; h3 = h3 * 5 + 0x96cd1c35;
			k4 *= C4; k4 = Integer.rotateLeft(k4, 18); k4 *= C1; h4 ^= k4;
			h4 = Integer.rotateLeft(h4, 13); h4 += h1; h4 = h4 * 5 + 0x32ac3b17;
		}

		int k1 = 0, k2 = 0, k3 = 0, k4 = 0;
		final int tail = nblocks * 16;
		switch (len & 15) {
		case 15: k4 ^= (data[tail + 14] & 0xff) << 16;
		case 14: k4 ^= (data[tail + 13] & 0xff) << 8;
		case 13: k4 ^= (data[tail + 12] & 0xff);
			k4 *= C4; k4 = Integer.rotateLeft(k4, 18); k4 *= C1; h4 ^= k4;
		case 12: k3 ^= (data[tail + 11] & 0xff) << 24;
		case 11: k3 ^= (data[tail + 10] & 0xff) << 16;
		case 10: k3 ^= (data[tail + 9] & 0xff) << 8;
		case 9: k3 ^= (data[tail + 8] & 0xff);
			k3 *= C3; k3 = Integer.rotateLeft(k3, 17); k3 *= C4; h3 ^= k3;
		case 8: k2 ^= (data[tail + 7] & 0xff) << 24;
		case 7: k2 ^= (data[tail + 6] & 0xff) << 16;
		case 6: k2 ^= (data[tail + 5] & 0xff) << 8;
		case 5: k2 ^= (data[tail + 4] & 0xff);
			k2 *= C2; k2 = Integer.rotateLeft(k2, 16); k2 *= C3; h2 ^= k2;
		case 4: k1 ^= (data[tail + 3] & 0xff) << 24;
		case 3: k1 ^= (data[tail + 2] & 0xff) << 16;
		case 2: k1 ^= (data[tail + 1] & 0xff) << 8;
		case 1: k1 ^= (data[tail] & 0xff);
			k1 *= C1; k1 = Integer.rotateLeft(k1, 15); k1 *= C2; h1 ^= k1;
		default:
			break;
		}

		h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;
		h1 += h2; h1 += h3; h1 += h4;
		h2 += h1; h3 += h1; h4 += h1;
		h1 = fmix32(h1); h2 = fmix32(h2); h3 = fmix32(h3); h4 = fmix32(h4);
		h1 += h2; h1 += h3; h1 += h4;
		h2 += h1; h3 += h1; h4 += h1;

		return (h1 & 0xffffffffL) | ((h2 & 0xffffffffL) << 32);
	}

	private static int fmix32(int h) {

		h ^= h >>> 16;
		h *= 0x85ebca6b;
		h ^= h >>> 13;
		h *= 0xc2b2ae35;
		h ^= h >>> 16;
		return h;
	}

	private static int le32(final byte[] data, final int i) {

		return (data[i] & 0xff)
				| ((data[i + 1] & 0xff) << 8)
				| ((data[i + 2] & 0xff) << 16)
				| ((data[i + 3] & 0xff) << 24);
	}
}
