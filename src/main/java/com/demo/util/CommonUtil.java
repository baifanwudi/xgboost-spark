package com.demo.util;

import scala.Tuple2;

/**
 * Java-HashMap转Scala-Immutable.Map
 *
 * @author AllenBai
 */
public class CommonUtil {
	@SuppressWarnings("unchecked")
	public static <K, V> scala.collection.immutable.Map<K, V> toScalaImmutableMap(java.util.Map<K, V> javaMap) {
		final java.util.List<scala.Tuple2<K, V>> list = new java.util.ArrayList<>(javaMap.size());
		for (final java.util.Map.Entry<K, V> entry : javaMap.entrySet()) {
			list.add(scala.Tuple2.<K, V>apply(entry.getKey(), entry.getValue()));
		}
		final scala.collection.Seq<Tuple2<K, V>> seq = scala.collection.JavaConverters.asScalaBufferConverter(list).asScala().toSeq();
		return (scala.collection.immutable.Map<K, V>) scala.collection.immutable.Map$.MODULE$.apply(seq);
	}
}

