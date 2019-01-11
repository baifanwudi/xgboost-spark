package com.demo.ml;

import com.demo.base.AbstractSparkSql;
import com.demo.base.HDFSFileSystem;
import com.demo.util.CommonUtil;
import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.dmg.pmml.PMML;
import org.jpmml.model.PMMLUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.jpmml.sparkml.PMMLBuilder;
//import org.shaded.dmg.pmml.PMML;

import javax.xml.bind.JAXBException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class PipelineModelTrain extends AbstractSparkSql {

    FileSystem fileSystem = HDFSFileSystem.fileSystem;

    @Override
    public void executeProgram(String[] args, SparkSession spark) throws IOException {

        Dataset<Row> trainData=spark.sql("select * from tmp_trafficwisdom.ml_train_data limit 50000");
//        trainData.show();
//        trainData.printSchema();
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

        CommonUtil.toScalaImmutableMap(javaMap);
        XGBoostEstimator xgBoostEstimator=new XGBoostEstimator( CommonUtil.<String,Object>toScalaImmutableMap(javaMap))
                .setFeaturesCol("features").setLabelCol("isclick");


        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler,xgBoostEstimator});
        PipelineModel pipelineModel = pipeline.fit(trainData);
        saveModel(pipelineModel);
        System.out.println(pipelineModel);

        try {
            savePMML(trainData.schema(), pipelineModel);
        } catch (JAXBException e) {
            e.printStackTrace();
        }

        trainData.unpersist();

    }



    public static void main(String[] args) throws IOException, JAXBException {

    PipelineModelTrain pipelineModelTrain=new PipelineModelTrain();
    pipelineModelTrain.runAll(args,true);
//        Dataset<Row> trainData = spark.read()
//                .option("inferschema", "true")
//                .option("header", "true")
//                .option("encoding", "gbk")
//                .csv("/Users/AllenBai/data/ml_offical_data.csv").drop("create_time");



//        StringIndexerModel labelIndexer = new StringIndexer()
//                .setInputCol("isgoumai")
//                .setOutputCol("label")
//                .fit(trainData);
//
//        StringIndexerModel trainTypeIndexer = new StringIndexer()
//                .setInputCol("traintype")
//                .setOutputCol("traintypeIndex")
//                .fit(trainData);
//        OneHotEncoder encoder = new OneHotEncoder()
//                .setInputCol("traintypeIndex")
//                .setOutputCol("traintypeVec");


//
//        VectorIndexer featureIndexer = new VectorIndexer()
//                .setInputCol("features")
//                .setOutputCol("indexedFeatures")
//                .setMaxCategories(10);
//
//        LogisticRegression lr = new LogisticRegression()
//                .setMaxIter(100)
//                .setRegParam(0.3)
//                .setElasticNetParam(0)
//                .setLabelCol("label")
//                .setFeaturesCol("features");
//                ;
//
////        GBTClassifier gbt = new GBTClassifier()
////                .setLabelCol("label")
////                .setFeaturesCol("indexedFeatures")
////                .setMaxIter(100);
//


//        StructType schema = ;


//        savePMML(schema,pipelineModel);
//        spark.stop();
//
//       byte[] result= new PMMLBuilder(trainData.schema(),pipelineModel).buildByteArray();
//        System.out.println(result);
//
//        Files.write(Paths.get("xgboost-pmml"),result);
//        new PMMLBuilder()
//        savePMML(trainData.schema(),pipelineModel);
    }

    private  void saveModel(PipelineModel pipelineModel) throws IOException {
//        pipelineModel.write().overwrite().save("file:///model/xgboost/pipemodel");
//        pipelineModel.write().overwrite().save("/data/twms/traffichuixing/model/orgin/pipemodel");
        pipelineModel.write().overwrite().save("/data/twms/traffichuixing/model/xgboost/");
    }

    private static void savePMML(StructType shcema, PipelineModel pipelineModel) throws IOException, JAXBException {
        PMML pmml = new PMMLBuilder(shcema, pipelineModel).build();
//        JAXBUtil.marshalPMML(pmml, new StreamResult(System.out));
        String targetFile = "/data/twms/traffichuixing/model/xgboost-pmml";
//        PMMLUtil.marshal(pmml, new FileOutputStream(targetFile));
        PMMLUtil.marshal(pmml, HDFSFileSystem.fileSystem.create(new Path(targetFile)));
    }


}
