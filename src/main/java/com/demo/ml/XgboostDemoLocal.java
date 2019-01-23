package com.demo.ml;

import com.demo.base.HDFSFileSystem;
import com.demo.util.CommonUtil;
import ml.dmlc.xgboost4j.java.XGBoostError;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator;
import org.apache.hadoop.fs.Path;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.PMML;
import org.jpmml.model.PMMLUtil;
import org.jpmml.sparkml.PMMLBuilder;

import javax.xml.bind.JAXBException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * @author allen
 * @date 2019/1/14.
 */
public class XgboostDemoLocal {

	public static void main(String[] args) throws IOException, XGBoostError {

		SparkSession spark =SparkSession.builder().master("local[*]").getOrCreate();

		Dataset<Row> trainData = spark.read()
				.option("inferschema", "true")
				.option("header", "true")
				.option("encoding", "gbk")
				.csv("/Users/AllenBai/data/ml_offical_data.csv").drop("create_time");

		trainData.printSchema();
		trainData.show(false);

		trainData.cache();
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
		Map<String,Object> javaMap=new HashMap<>();
		javaMap.put("objective","binary:logistic");
		javaMap.put("eta",0.1);
		javaMap.put("max_depth",9);
		javaMap.put("min_child_weight",5);
		javaMap.put("alpha",1);
		javaMap.put("eval_metric","logloss");
		javaMap.put("num_round","20");
		javaMap.put("missing",-1);

		CommonUtil.toScalaImmutableMap(javaMap);
		XGBoostEstimator xgBoostEstimator=new XGBoostEstimator( CommonUtil.<String,Object>toScalaImmutableMap(javaMap))
				.setFeaturesCol("features").setLabelCol("isclick");

		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[]{assembler,xgBoostEstimator});

		PipelineModel pipelineModel = pipeline.fit(trainData);

		Dataset<Row> predictResult=pipelineModel.transform(trainData);
		predictResult.show(false);

		XGBoostClassificationModel xgBoostClassificationModel=(XGBoostClassificationModel) (pipelineModel.stages()[1]);

		//evaluate
		BinaryClassificationEvaluator evaluator=new BinaryClassificationEvaluator().setLabelCol("isclick").setRawPredictionCol("probabilities");
		Double aucArea=evaluator.evaluate(predictResult);
		System.out.println("auc is :"+aucArea);

//		pipelineModel.write().overwrite().save("file:///model/xgboost/pipemodel");

	}

	private static void savePMML(StructType shcema, PipelineModel pipelineModel) throws IOException, JAXBException {
		PMML pmml = new PMMLBuilder(shcema, pipelineModel).build();
//        JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
		String targetFile = "/data/twms/traffichuixing/model/xgboost-pmml";
//        PMMLUtil.marshal(pmml, new FileOutputStream(targetFile));
		PMMLUtil.marshal(pmml, HDFSFileSystem.fileSystem.create(new Path(targetFile)));
	}

}
