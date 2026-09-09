package org.janelia.scicomp.n5.precomputed;

import org.janelia.saalfeldlab.n5.DataType;

/**
 * Maps Neuroglancer precomputed {@code data_type} strings to the N5
 * {@link DataType}. Precomputed data is always little-endian.
 * <p>
 * The precomputed {@code data_type} labels ({@code uint8}, {@code int8},
 * {@code uint16}, {@code int16}, {@code uint32}, {@code int32}, {@code uint64},
 * {@code float32}) coincide with the N5 {@link DataType#toString() labels}, so
 * the mapping is a direct {@link DataType#fromString(String)} lookup.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedDataType {

	private PrecomputedDataType() {}

	/**
	 * Returns the N5 {@link DataType} for a precomputed {@code data_type}
	 * string, or {@code null} if unknown.
	 *
	 * @param dataType the precomputed data_type string
	 * @return the N5 data type
	 */
	public static DataType fromString(final String dataType) {

		return dataType == null ? null : DataType.fromString(dataType.toLowerCase());
	}

	/**
	 * Number of bytes per element of the given data type.
	 *
	 * @param dataType the data type
	 * @return the number of bytes per element
	 */
	public static int bytesPerElement(final DataType dataType) {

		switch (dataType) {
		case UINT8:
		case INT8:
		case OBJECT:
			return 1;
		case UINT16:
		case INT16:
			return 2;
		case UINT32:
		case INT32:
		case FLOAT32:
			return 4;
		case UINT64:
		case INT64:
		case FLOAT64:
			return 8;
		default:
			throw new IllegalArgumentException("Unsupported precomputed data type: " + dataType);
		}
	}
}
