package com.demo.cross;

import com.demo.base.AbstractSparkSql;
import com.demo.util.CommonUtil;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator;
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
		javaMap.put("eta",0.1);
		javaMap.put("max_depth",9);
		javaMap.put("min_child_weight",5);
		//L2
		javaMap.put("lambda",0.2);
		javaMap.put("eval_metric","logloss");
		javaMap.put("num_round","150");
		javaMap.put("missing",-99);
		//default 1,提高并行度
		javaMap.put("nworkers",32);
		javaMap.put("numEarlyStoppingRounds",20);
		javaMap.put("checkpoint_path","/data/twms/traffichuixing/checkpoint/xgboost");
		CommonUtil.toScalaImmutableMap(javaMap);
		XGBoostEstimator xgBoostEstimator=new XGBoostEstimator( CommonUtil.<String,Object>toScalaImmutableMap(javaMap))
				.setFeaturesCol("features").setLabelCol("isclick");


		BinaryClassificationEvaluator evaluator=new BinaryClassificationEvaluator().setLabelCol("isclick").setRawPredictionCol("probabilities");


		ParamMap[] paramGrid=new ParamGridBuilder()
				.addGrid(xgBoostEstimator.maxDepth(),new int[]{4,6,8})
				.addGrid(xgBoostEstimator.round(),new int[]{40,70,100})
//				.addGrid(xgBoostEstimator.eta(),new double[]{0.1,0.2,0.3})
//				.addGrid(xgBoostEstimator.lambda(),new double[]{0.1,0.3,0.6})
//				.addGrid(xgBoostEstimator.minChildWeight(),new double[]{1.0,3.0,5.0})
				.build();

		CrossValidator crossValidator=new CrossValidator().setEstimator(xgBoostEstimator)
				.setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(4);

		CrossValidatorModel cvModel=crossValidator.fit(trainData);

		XGBoostClassificationModel xgBestModel=(XGBoostClassificationModel)(cvModel.bestModel());
		ParamMap[] paramMaps=cvModel.getEstimatorParamMaps();
		double[] aucAreas=cvModel.avgMetrics();

		for(int i=0;i<paramMaps.length;i++){
			System.out.println("--------------"+i+"----------------");
			System.out.println("param:"+paramMaps[i]);
			System.out.println("auc metric is :"+aucAreas[i]);
		}

		System.out.println("---------best param is :---------------");
		System.out.println(xgBestModel.extractParamMap());


		trainData.unpersist();
//
//		Dataset<Row> testData=assembler.transform(spark.sql("select * from tmp_trafficwisdom.ml_test_data "));
//		Dataset<Row> testResult=xgBestModel.transform(testData);
//
//		Double aucArea=evaluator.evaluate(testResult);
//		System.out.println("------------------test result predict----------------");
//		System.out.println("auc is :"+aucArea);


	}

	public static void main(String[] args) throws IOException {
		XGBoostCrossDemo xgboostCrossDemo=new XGBoostCrossDemo();
		xgboostCrossDemo.runAll(args,true);
	}
}
