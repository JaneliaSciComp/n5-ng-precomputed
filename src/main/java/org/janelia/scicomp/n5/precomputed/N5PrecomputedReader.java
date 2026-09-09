package org.janelia.scicomp.n5.precomputed;

import org.janelia.saalfeldlab.n5.FileSystemKeyValueAccess;
import org.janelia.saalfeldlab.n5.N5Exception;

import com.google.gson.GsonBuilder;

/**
 * A {@link PrecomputedKeyValueReader} over the local filesystem.
 * <p>
 * The "just give me a path" entry point. For cloud backends (S3, GCS, ...)
 * construct a {@link PrecomputedKeyValueReader} directly with the appropriate
 * {@code KeyValueAccess}.
 *
 * @author Stephan Preibisch
 */
public class N5PrecomputedReader extends PrecomputedKeyValueReader {

	/**
	 * Opens an {@link N5PrecomputedReader} at a given base path.
	 *
	 * @param basePath    the precomputed container base path
	 * @param gsonBuilder a gson builder
	 * @param cacheMeta   cache the parsed {@code info} document
	 * @throws N5Exception if the container cannot be read or does not exist
	 */
	public N5PrecomputedReader(final String basePath, final GsonBuilder gsonBuilder, final boolean cacheMeta) throws N5Exception {

		super(new FileSystemKeyValueAccess(), basePath, gsonBuilder, cacheMeta);
	}

	/**
	 * Opens an {@link N5PrecomputedReader} at a given base path.
	 *
	 * @param basePath  the precomputed container base path
	 * @param cacheMeta cache the parsed {@code info} document
	 * @throws N5Exception if the container cannot be read or does not exist
	 */
	public N5PrecomputedReader(final String basePath, final boolean cacheMeta) throws N5Exception {

		this(basePath, new GsonBuilder(), cacheMeta);
	}

	/**
	 * Opens an {@link N5PrecomputedReader} at a given base path.
	 *
	 * @param basePath the precomputed container base path
	 * @throws N5Exception if the container cannot be read or does not exist
	 */
	public N5PrecomputedReader(final String basePath) throws N5Exception {

		this(basePath, new GsonBuilder(), true);
	}
}
