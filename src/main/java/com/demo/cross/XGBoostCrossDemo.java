package com.demo.cross;

import com.demo.base.AbstractSparkSql;
import com.demo.util.CommonUtil;
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
 * @date 2019/1/21.
 */
public class XGBoostCrossDemo extends AbstractSparkSql {

	@Override
	public void executeProgram(String[] args, SparkSession spark) throws IOException {

		Dataset<Row> tableData=spark.sql("select * from tmp_trafficwisdom.ml_train_data where future_day>=0 ")
				.drop("userid,city,from_place,to_place,start_city_name,end_city_name,start_city_id,end_city_id".split(","));

		String[] features=new String[]{
				"category", "future_day",
				"banner_min_time","banner_min_price",
				"page_train", "page_flight", "page_bus",
				"page_transfer",
				"start_end_distance", "total_transport", "high_railway_percent", "avg_time", "min_time",
				"avg_price", "min_price",
				"label_05060801", "label_05060701", "label_05060601", "label_02050601", "label_02050501", "label_02050401",
				"is_match_category", "train_consumer_prefer", "flight_consumer_prefer", "bus_consumer_prefer"
		};

		VectorAssembler assembler = new VectorAssembler().setInputCols(features).setOutputCol("features");
		Dataset<Row> trainData=assembler.transform(tableData).select("features","isclick");
		trainData.cache();

		Map<String,Object> javaMap=new HashMap<>(12);
		javaMap.put("objective","binary:logistic");
		//learning_rate
		javaMap.put("eta",0.25);
		javaMap.put("max_depth",9);
		javaMap.put("min_child_weight",1);
		//L2
		javaMap.put("lambda",0.0001);
		javaMap.put("eval_metric","logloss");
		javaMap.put("num_round","70");
		javaMap.put("missing",-99);
		javaMap.put("num_early_stopping_rounds",20);
		javaMap.put("maximize_evaluation_metrics",false);
		//default 1,提高并行度
		javaMap.put("num_workers",24);
		javaMap.put("silent",1);
		//注意0.81有一个bug，必须设置seed为long类型，要不保存model会报错。
		javaMap.put("seed",2019L);
		CommonUtil.toScalaImmutableMap(javaMap);
		XGBoostClassifier xgBoostClassifier=new XGBoostClassifier( CommonUtil.<String,Object>toScalaImmutableMap(javaMap))
				.setFeaturesCol("features").setLabelCol("isclick").setProbabilityCol("probabilities");
		BinaryClassificationEvaluator evaluator=new BinaryClassificationEvaluator().setLabelCol("isclick").setRawPredictionCol("probabilities");

		/**
		 * max_depth:9
		 * eta:0.25
		 * lambda:0.0001
		 * num_round:65
		 * min_child_weight:1
		 */
		ParamMap[] paramGrid=new ParamGridBuilder()
//				.addGrid(xgBoostClassifier.maxDepth(),new int[]{7,8,9})
				.addGrid(xgBoostClassifier.numRound(),new int[]{65,70,75})
				.addGrid(xgBoostClassifier.eta(),new double[]{0.25,0.3,0.35})
				.addGrid(xgBoostClassifier.lambda(),new double[]{0.001,0.02,0.05})
//				.addGrid(xgBoostClassifier.minChildWeight(),new double[]{1.0,3.0,5.0})
				.build();

		CrossValidator crossValidator=new CrossValidator().setEstimator(xgBoostClassifier)
				.setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(5);

		CrossValidatorModel cvModel=crossValidator.fit(trainData);


		XGBoostClassificationModel xgBestModel=(XGBoostClassificationModel)(cvModel.bestModel());
		ParamMap[] paramMaps=cvModel.getEstimatorParamMaps();
		double[] aucAreas=cvModel.avgMetrics();

		System.out.println("------------------------------------");
		for(int i=0;i<paramMaps.length;i++){
			System.out.println("--------------"+i+"----------------");
			System.out.println("param:"+paramMaps[i]);
			System.out.println("auc metric is :"+aucAreas[i]);
		}

		System.out.println("---------best param is :---------------");
		System.out.println(xgBestModel.extractParamMap());

		System.out.println("--------------------------------------");
		trainData.unpersist();

		cvModel.write().overwrite().save("/data/twms/traffichuixing/model/cross-xgboost");
		xgBestModel.write().overwrite().save("/data/twms/traffichuixing/model/xgboost/");

//		Dataset<Row> testData=assembler.transform(spark.sql("select * from tmp_trafficwisdom.ml_test_data "));
//		Dataset<Row> testResult=xgBestModel.transform(testData);
//
//		Double aucArea=evaluator.evaluate(testResult);
//		System.out.println("------------------test result predict----------------");
//		System.out.println("auc is :"+aucArea);
//		System.out.println("---------------------------------------------------");

	}

	public static void main(String[] args) throws IOException {
		XGBoostCrossDemo xgboostCrossDemo=new XGBoostCrossDemo();
		xgboostCrossDemo.runAll(args,true);
	}
}
