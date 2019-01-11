package com.demo.base;

import org.apache.spark.sql.Dataset;
import scala.Option;
import scala.collection.JavaConversions;
import scala.collection.Seq;

import java.util.Arrays;
import java.util.List;

/**
 * @author allen
 * Created by allen on 25/07/2017.
 */
public final class ColumnUtil {

	/**
	 * 将list<String>转换为Scala Seq<String>,dataset join使用
	 * @param listName
	 * @return
	 */
	public static Option<Seq<String>> columnNames(List<String> listName){
		Seq<String> b=(Seq<String>) listName;
		return Option.<Seq<String>>apply(b);
	}

	/**
	 * 将String切分为转换为Scala Seq<String>,dataset join使用
	 * @param columnsName
	 * @return
	 */
	public static Seq<String> columnNames(String columnsName){
		List<String> list= Arrays.asList(columnsName.split(","));
		return JavaConversions.<String>asScalaBuffer(list);
	}

	/**
	 * 判断dataset是否含某个column
	 * @param dataset
	 * @param colName
	 * @return
	 */
	public static boolean hasColumn(Dataset dataset, String colName){
		return Arrays.asList(dataset.columns()).contains(colName);
	}
}
