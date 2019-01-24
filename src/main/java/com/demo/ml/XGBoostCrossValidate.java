package com.demo.ml;

import com.demo.util.CommonUtil;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author allen
 * @date 2019/1/14.
 */
public class XGBoostCrossValidate {

	public static void main(String[] args) throws IOException, XGBoostError {

		SparkSession spark =SparkSession.builder().master("local[*]").getOrCreate();

		Dataset<Row> tableData = spark.read()
				.option("inferschema", "true")
				.option("header", "true")
				.option("encoding", "gbk")
				.csv("/Users/AllenBai/data/ml_offical_data.csv").drop("create_time");

		tableData.printSchema();
		tableData.show(false);


		String[] features=new String[]{
				"category", "future_day",
				"banner_min_time","banner_min_price",
				"page_train", "page_flight", "page_bus",
				"page_transfer",
				"start_end_distance", "total_transport", "high_railway_percent", "avg_time", "min_time",
				"avg_price", "min_price",
				"label_05060801", "label_05060701", "label_05060601", "label_02050601", "label_02050501", "label_02050401",
				"is_match_category", "train_consumer_prefer", "flight_consumer_prefer"
				, "bus_consumer_prefer"
		};

		VectorAssembler assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");

		Dataset<Row> trainData=assembler.transform(tableData).select("features","isclick");
		trainData.cache();
		trainData.printSchema();
		trainData.show(false);
		Map<String,Object> javaMap=new HashMap<>();
		javaMap.put("objective","binary:logistic");
		javaMap.put("eta",0.1);
		javaMap.put("max_depth",9);
		javaMap.put("min_child_weight",5);
		javaMap.put("lambda",0.2);
		javaMap.put("eval_metric","logloss");
		javaMap.put("num_round","20");
		javaMap.put("missing",-1);


		CommonUtil.toScalaImmutableMap(javaMap);

		XGBoostClassifier xgBoostEstimator=new XGBoostClassifier( CommonUtil.<String,Object>toScalaImmutableMap(javaMap))
				.setFeaturesCol("features").setLabelCol("isclick");

        //evaluate
		BinaryClassificationEvaluator evaluator=new BinaryClassificationEvaluator().setLabelCol("isclick").setRawPredictionCol("probabilities");


		ParamMap[] paramGrid=new ParamGridBuilder()
				.addGrid(xgBoostEstimator.maxDepth(),new int[]{4,5})
				.addGrid(xgBoostEstimator.eta(),new double[]{0.2,0.3})
				.build();

		CrossValidator crossValidator=new CrossValidator().setEstimator(xgBoostEstimator)
				.setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3);

		CrossValidatorModel cvModel=crossValidator.fit(trainData);

		XGBoostClassificationModel xgBestModel=(XGBoostClassificationModel)(cvModel.bestModel());

		ParamMap[] paramMaps=cvModel.getEstimatorParamMaps();
		double[] aucArea=cvModel.avgMetrics();

		for(int i=0;i<paramMaps.length;i++){
			System.out.println("--------------"+i+"----------------");
			System.out.println("param:"+paramMaps[i]);
			System.out.println("auc metric is :"+aucArea[i]);
		}

		System.out.println("---------best param is :---------------");
		System.out.println(xgBestModel.extractParamMap());

		trainData.unpersist();

	}
}
