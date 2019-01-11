package com.demo.base;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;

public abstract class AbstractSparkSql {

    protected Logger logger = LoggerFactory.getLogger(this.getClass());

    /**
     * 将params注入sparkconf
     *
     * @param params
     * @return
     */
    public static SparkConf addParams(SparkConf sparkConf, Map<String, String> params) {
        params.forEach((k, v) -> {
            sparkConf.set(k, v);
        });
        return sparkConf;
    }

    /**
     * 新建sparksession
     *
     * @param isHiveSupported
     * @param isEsSupported
     * @return
     */
    protected SparkSession createSparkSession(Boolean isHiveSupported, Boolean isEsSupported) {
        SparkSession spark;
        SparkConf sparkConf = new SparkConf();
        String appName = this.getClass().getSimpleName();
        //默认读取spark配置
        sparkConf = addParams(sparkConf, ConfigUtil.getSparkParams());
        //读取hive配置
        if (isHiveSupported) {
            sparkConf = addParams(sparkConf, ConfigUtil.getHiveParams());
            logger.info("spark read hive config");
        }
        //读取es配置
        if (isEsSupported) {
            sparkConf = addParams(sparkConf, ConfigUtil.getEsParams());
            logger.info("spark read es config");
        }
        if (isHiveSupported) {
            spark = SparkSession.builder().appName(appName).config(sparkConf).enableHiveSupport().getOrCreate();
            logger.info("spark enable hive, begin to execute the program");
        } else {
            spark = SparkSession.builder().appName(appName).config(sparkConf).getOrCreate();
            logger.info("spark begin to execute the program");
        }
        return spark;
    }

    /**
     * spark初始化,如果激活hive,需要切换数据库.
     *
     * @param args            格式为2018-05-24
     * @param isHiveSupported
     * @param isEsSupported
     * @throws IOException
     */
    protected void executeSpark(String[] args, Boolean isHiveSupported, Boolean isEsSupported) throws IOException {
        SparkSession spark = createSparkSession(isHiveSupported, isEsSupported);
        executeProgram(args, spark);
        logger.info("spark has finished the program ");
		spark.stop();
    }

    /**
     * @param args
     * @param isHiveSupported
     * @param isEsSupported
     * @throws IOException
     */
    public void runAll(String[] args, Boolean isHiveSupported, Boolean isEsSupported) throws IOException {
        executeSpark(args, isHiveSupported, isEsSupported);
    }

    /**
     * 是否激活hive
     *
     * @param args
     * @param isHiveSupported
     * @throws IOException
     */
    public void runAll(String[] args, Boolean isHiveSupported) throws IOException {
        runAll(args, isHiveSupported, false);
    }

    /**
     * 只激活hive
     *
     * @param args
     * @throws IOException
     */
    public void runAll(String [] args) throws IOException {
        runAll(args, true, false);
    }

    /**
     * spark 执行内容
     *
     * @param args  格式为2018-05-24
     * @param spark
     * @throws IOException
     */
    public abstract void executeProgram(String[] args, SparkSession spark) throws IOException;

}
