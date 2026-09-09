package org.janelia.scicomp.n5.precomputed;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPInputStream;

import org.janelia.saalfeldlab.n5.CachedGsonKeyValueN5Reader;
import org.janelia.saalfeldlab.n5.DataBlock;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.GsonKeyValueN5Reader;
import org.janelia.saalfeldlab.n5.GsonUtils;
import org.janelia.saalfeldlab.n5.KeyValueAccess;
import org.janelia.saalfeldlab.n5.N5Exception;
import org.janelia.saalfeldlab.n5.N5URI;
import org.janelia.saalfeldlab.n5.cache.N5JsonCache;
import org.janelia.saalfeldlab.n5.cache.N5JsonCacheableContainer;
import org.janelia.scicomp.n5.precomputed.sharding.PrecomputedShardReader;
import org.janelia.saalfeldlab.n5.readdata.VolatileReadData;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;

/**
 * A read-only {@code N5Reader} for Neuroglancer precomputed volumes over any
 * {@link KeyValueAccess} backend.
 * <p>
 * A precomputed volume centralizes all metadata in a single {@code info}
 * document at the container root. This reader parses (and caches) that document
 * and presents:
 * <ul>
 * <li>the container root {@code "/"} as an N5 group whose attributes are the
 * {@code info} JSON;</li>
 * <li>each {@code scales[].key} as an N5 dataset ({@code [x, y, z, channel]}).</li>
 * </ul>
 * Unsharded chunk reads use the inherited {@link GsonKeyValueN5Reader} block
 * pipeline (via {@link PrecomputedDatasetAttributes#relativeBlockPath}); sharded
 * scales are routed to {@link PrecomputedShardReader}.
 *
 * @author Stephan Preibisch
 */
public class PrecomputedKeyValueReader implements CachedGsonKeyValueN5Reader, N5JsonCacheableContainer {

	public static final Version VERSION = new Version(1, 0, 0);

	public static final String INFO_FILE = "info";

	protected final KeyValueAccess keyValueAccess;

	protected final Gson gson;

	protected final boolean cacheMeta;

	protected URI uri;

	private transient PrecomputedInfo info;
	private transient JsonObject infoJson;
	private transient boolean infoLoaded = false;

	/**
	 * Opens a {@link PrecomputedKeyValueReader}.
	 *
	 * @param keyValueAccess the storage backend
	 * @param basePath       the container base path
	 * @param gsonBuilder    a gson builder
	 * @param cacheMeta      cache the parsed {@code info} document (the info is
	 *                       always parsed at most once regardless; changes by an
	 *                       independent writer are not tracked)
	 * @throws N5Exception if the container does not exist or cannot be read
	 */
	public PrecomputedKeyValueReader(
			final KeyValueAccess keyValueAccess,
			final String basePath,
			final GsonBuilder gsonBuilder,
			final boolean cacheMeta) throws N5Exception {

		this(keyValueAccess, basePath, gsonBuilder, cacheMeta, true);
	}

	protected PrecomputedKeyValueReader(
			final KeyValueAccess keyValueAccess,
			final String basePath,
			final GsonBuilder gsonBuilder,
			final boolean cacheMeta,
			final boolean checkRootExists) throws N5Exception {

		this.keyValueAccess = keyValueAccess;
		this.gson = gsonBuilder.disableHtmlEscaping().create();
		this.cacheMeta = cacheMeta;

		try {
			uri = keyValueAccess.uri(basePath);
		} catch (final URISyntaxException e) {
			throw new N5Exception(e);
		}

		if (checkRootExists && !exists("/"))
			throw new N5Exception.N5IOException("No precomputed container (info) exists at " + basePath);
	}

	// -- precomputed info -----------------------------------------------------

	/**
	 * Returns the parsed {@code info} document, or {@code null} if there is no
	 * {@code info} file at the container root.
	 *
	 * @return the parsed info or {@code null}
	 */
	public PrecomputedInfo getInfo() {

		if (!infoLoaded || !cacheMeta) {
			final String infoPath = keyValueAccess.compose(uri, INFO_FILE);
			if (!keyValueAccess.isFile(infoPath)) {
				info = null;
				infoJson = null;
			} else {
				try (final VolatileReadData rd = keyValueAccess.createReadData(infoPath)) {
					// Some backends (e.g. GCS with Content-Encoding: gzip) return the
					// info gzip-compressed on ranged/channel reads; decompress if so.
					final String json = new String(maybeGunzip(rd.allBytes()), StandardCharsets.UTF_8);
					final JsonElement el = gson.fromJson(json, JsonElement.class);
					infoJson = el != null && el.isJsonObject() ? el.getAsJsonObject() : null;
					info = gson.fromJson(el, PrecomputedInfo.class);
				} catch (final N5Exception.N5NoSuchKeyException e) {
					info = null;
					infoJson = null;
				}
			}
			infoLoaded = true;
		}
		return info;
	}

	private JsonObject getInfoJson() {

		getInfo();
		return infoJson;
	}

	// -- N5Reader surface backed by the info document -------------------------

	@Override
	public boolean exists(final String pathName) {

		final String normal = N5URI.normalizeGroupPath(pathName);
		if (getInfo() == null)
			return false;
		return normal.isEmpty() || getInfo().getScale(normal) != null;
	}

	@Override
	public boolean groupExists(final String pathName) {

		final String normal = N5URI.normalizeGroupPath(pathName);
		return normal.isEmpty() && getInfo() != null;
	}

	@Override
	public boolean datasetExists(final String pathName) throws N5Exception {

		final String normal = N5URI.normalizeGroupPath(pathName);
		return getInfo() != null && getInfo().getScale(normal) != null;
	}

	@Override
	public DatasetAttributes getDatasetAttributes(final String pathName) throws N5Exception {

		final PrecomputedInfo info = getInfo();
		if (info == null)
			return null;
		final PrecomputedInfo.Scale scale = info.getScale(N5URI.normalizeGroupPath(pathName));
		return scale == null ? null : new PrecomputedDatasetAttributes(info, scale);
	}

	@Override
	public JsonElement getAttributes(final String pathName) throws N5Exception {

		final PrecomputedInfo info = getInfo();
		if (info == null)
			return null;

		final String normal = N5URI.normalizeGroupPath(pathName);
		if (normal.isEmpty())
			return getInfoJson();

		final PrecomputedInfo.Scale scale = info.getScale(normal);
		if (scale == null)
			return null;

		final JsonElement scaleTree = gson.toJsonTree(scale);
		if (!scaleTree.isJsonObject())
			return scaleTree;
		final JsonObject obj = scaleTree.getAsJsonObject();
		obj.addProperty("data_type", info.dataType);
		obj.addProperty("num_channels", info.numChannels);
		if (info.type != null)
			obj.addProperty("type", info.type);
		return obj;
	}

	@Override
	public <T> T getAttribute(final String pathName, final String key, final Class<T> clazz) throws N5Exception {

		final JsonElement attributes = getAttributes(pathName);
		try {
			return GsonUtils.readAttribute(attributes, N5URI.normalizeAttributePath(key), clazz, gson);
		} catch (JsonSyntaxException | NumberFormatException | ClassCastException e) {
			throw new N5Exception.N5ClassCastException(e);
		}
	}

	@Override
	public <T> T getAttribute(final String pathName, final String key, final Type type) throws N5Exception {

		final JsonElement attributes = getAttributes(pathName);
		try {
			return GsonUtils.readAttribute(attributes, N5URI.normalizeAttributePath(key), type, gson);
		} catch (JsonSyntaxException | NumberFormatException | ClassCastException e) {
			throw new N5Exception.N5ClassCastException(e);
		}
	}

	@Override
	public String[] list(final String pathName) throws N5Exception {

		final PrecomputedInfo info = getInfo();
		if (info == null)
			return new String[0];
		final String normal = N5URI.normalizeGroupPath(pathName);
		return normal.isEmpty() ? info.getScaleKeys() : new String[0];
	}

	@Override
	public Version getVersion() throws N5Exception {

		return VERSION;
	}

	// -- block reading --------------------------------------------------------

	@Override
	public <T> DataBlock<T> readBlock(
			final String pathName,
			final DatasetAttributes datasetAttributes,
			final long... gridPosition) throws N5Exception {

		final DatasetAttributes converted = getConvertedDatasetAttributes(datasetAttributes);
		if (converted instanceof PrecomputedDatasetAttributes
				&& ((PrecomputedDatasetAttributes)converted).isShardedPrecomputed()) {
			return PrecomputedShardReader.readBlock(keyValueAccess, uri,
					N5URI.normalizeGroupPath(pathName), (PrecomputedDatasetAttributes)converted, gridPosition);
		}
		return CachedGsonKeyValueN5Reader.super.readBlock(pathName, datasetAttributes, gridPosition);
	}

	@Override
	public <T> DataBlock<T> readChunk(
			final String pathName,
			final DatasetAttributes datasetAttributes,
			final long... gridPosition) throws N5Exception {

		final DatasetAttributes converted = getConvertedDatasetAttributes(datasetAttributes);
		if (converted instanceof PrecomputedDatasetAttributes
				&& ((PrecomputedDatasetAttributes)converted).isShardedPrecomputed()) {
			return PrecomputedShardReader.readBlock(keyValueAccess, uri,
					N5URI.normalizeGroupPath(pathName), (PrecomputedDatasetAttributes)converted, gridPosition);
		}
		return CachedGsonKeyValueN5Reader.super.readChunk(pathName, datasetAttributes, gridPosition);
	}

	// -- accessors ------------------------------------------------------------

	@Override
	public Gson getGson() {

		return gson;
	}

	@Override
	public KeyValueAccess getKeyValueAccess() {

		return keyValueAccess;
	}

	@Override
	public URI getURI() {

		return uri;
	}

	@Override
	public String getAttributesKey() {

		return INFO_FILE;
	}

	@Override
	public DatasetAttributes createDatasetAttributes(final JsonElement attributes) {

		// precomputed dataset attributes require the info + scale context;
		// they are built in getDatasetAttributes, not from a bare JsonElement.
		return null;
	}

	// -- caching (this reader caches the info document itself) ----------------

	@Override
	public boolean cacheMeta() {

		return false;
	}

	@Override
	public N5JsonCache getCache() {

		return null;
	}

	@Override
	public N5JsonCache newCache() {

		return null;
	}

	// -- N5JsonCacheableContainer (not used since cacheMeta() is false) -------

	@Override
	public JsonElement getAttributesFromContainer(final String normalPathName, final String normalCacheKey) {

		return getAttributes(normalPathName);
	}

	@Override
	public boolean existsFromContainer(final String normalPathName, final String normalCacheKey) {

		return exists(normalPathName);
	}

	@Override
	public boolean isGroupFromContainer(final String normalPathName) {

		return groupExists(normalPathName);
	}

	@Override
	public boolean isDatasetFromContainer(final String normalPathName) {

		return datasetExists(normalPathName);
	}

	@Override
	public boolean isGroupFromAttributes(final String normalCacheKey, final JsonElement attributes) {

		return true;
	}

	@Override
	public boolean isDatasetFromAttributes(final String normalCacheKey, final JsonElement attributes) {

		return false;
	}

	@Override
	public String[] listFromContainer(final String normalPathName) {

		return list(normalPathName);
	}

	@Override
	public String toString() {

		return String.format("%s[access=%s, basePath=%s]", getClass().getSimpleName(), keyValueAccess, uri.getPath());
	}

	private static byte[] maybeGunzip(final byte[] bytes) {

		if (bytes.length < 2 || (bytes[0] & 0xff) != 0x1f || (bytes[1] & 0xff) != 0x8b)
			return bytes;
		try (final GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
			final ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, bytes.length * 3));
			final byte[] buf = new byte[8192];
			int r;
			while ((r = in.read(buf)) > 0)
				out.write(buf, 0, r);
			return out.toByteArray();
		} catch (final IOException e) {
			throw new N5Exception.N5IOException("failed to gunzip info", e);
		}
	}
}
